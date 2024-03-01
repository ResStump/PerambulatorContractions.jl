import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path

include("allocate_arrays.jl")


@doc raw"""
    Parms

Important parameters for the perambulator contractions.
"""
struct Parms
    # String containt in parameter toml file passed to program
    parms_toml_string::String

    # Paths
    perambulator_dir
    mode_doublets_dir
    sparse_modes_dir
    result_dir

    # Configuration numbers and source times
    cnfg_indices::Vector{Int}
    tsrc_arr::Array{Int, 2}

    # Lattice size in time and space, and number of modes
    Nₜ::Int
    Nₖ::Vector{Int} # spatial components k = 1, 2, 3
    N_modes::Int

    # Number of configurations and sources
    N_cnfg::Int
    N_src::Int

    # Momentum
    p::Vector{Int}
end


@doc raw"""
    read_parameters()

Read the parameters stored in the parameter file passed to the program with the flag -i and
return the dictonary `parms_toml` and the Parms instance `parms`.
"""
function read_parameters()
    # Search for parameter file in arguments passed to program
    parms_file_index = findfirst(arg -> arg == "-i", ARGS)
    if isnothing(parms_file_index)
        throw(ArgumentError("argument -i not provided to the program."))
    elseif parms_file_index == size(ARGS)[1]
        throw(ArgumentError("argument after -i not provided to the program."))
    end

    parms_file = ARGS[parms_file_index+1]

    # Read parameters from parameter file and store them in the `parms_toml`
    parms_toml_string = read(parms_file, String)
    parms_toml = TOML.parse(parms_toml_string)

    # Read source times
    tsrc_list = DelimitedFiles.readdlm(
        parms_toml["Directories and Files"]["tsrc_list"], ' ', Int
    )
    
    # Set paths
    perambulator_dir = Path(parms_toml["Directories and Files"]["perambulator_dir"])
    mode_doublets_dir = Path(parms_toml["Directories and Files"]["mode_doublets_dir"])
    sparse_modes_dir = Path(parms_toml["Directories and Files"]["sparse_modes_dir"])
    result_dir = Path(parms_toml["Directories and Files"]["result_dir"])

    # Lattice size
    Nₜ = parms_toml["Geometry"]["N_t"]
    Nₖ = parms_toml["Geometry"]["N_k"]

    # Set `cnfg_indices`
    first_cnfg = parms_toml["Configurations"]["first"]
    step_cnfg = parms_toml["Configurations"]["step"]
    last_cnfg = parms_toml["Configurations"]["last"]
    N_cnfg = (last_cnfg - first_cnfg) ÷ step_cnfg + 1
    cnfg_indices = Array(first_cnfg:step_cnfg:last_cnfg)

    # Find first source times for specified configurations
    tsrc_first = Array{Int, 1}(undef, N_cnfg)
    for (i_cnfg, n_cnfg) in enumerate(cnfg_indices)
        idx = findfirst(n -> n == n_cnfg, tsrc_list[:, 1])
        tsrc_first[i_cnfg] = tsrc_list[idx, 2]
    end

    # Store all source times in `tsrc_arr`
    N_src = parms_toml["Sources"]["N_src"]
    src_separation = parms_toml["Sources"]["src_separation"]
    tsrc_arr = hcat([tsrc_first .+ i_src*src_separation
                     for i_src in 0:N_src-1]...)
    tsrc_arr = mod.(tsrc_arr, Nₜ) # periodically shift values >= Nₜ

    # Extract number of modes from one perambulator file
    perambulator_file = 
        "perambulator_$(parms_toml["Perambulator"]["label_base"])" * "$(tsrc_list[1, 2])_" *
        "$(parms_toml["Run name"]["name"])n$(tsrc_list[1, 1])"
    
    file = HDF5.h5open(string(perambulator_dir/perambulator_file), "r")
    N_modes = size(file["perambulator"])[4]
    close(file)

    # Momentum
    p = parms_toml["Momentum"]["p"]

    # Store all parameters
    parms = Parms(parms_toml_string, perambulator_dir, mode_doublets_dir,
        sparse_modes_dir, result_dir, cnfg_indices, tsrc_arr, Nₜ, Nₖ,
        N_modes, N_cnfg, N_src, p)
    
    return parms, parms_toml
end

@doc raw"""
    read_perambulator!(perambulator_file, τ_αkβlt)

Read the perambulator from the HDF5 file `perambulator_file` and store it in `τ_αkβlt`.

### Indices
The indices of `τ_αkβlt` are:
- α: sink spinor
- k: sink Laplace mode
- β: source spinor
- l: source Laplace mode
- t: sink time

See also: `read_perambulator`.
"""
function read_perambulator!(perambulator_file, τ_αkβlt)
    # Check if shape is correct
    N_color1, N_modes1, N_color2, N_modes2, Nₜ = size(τ_αkβlt)
    if N_color1 != N_color2 != 4
        throw(DimensionMismatch("dimensions of spinor axis in perambulator don't match."))
    end
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("number of modes don't match."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimensions of time axis don't match."))
    end
    
    # Read perambulator and set noise index to 1 (since noise is `ones`)
    hdf5_file = HDF5.h5open(string(perambulator_file), "r")
    τ_αkβlt[:] = hdf5_file["perambulator"][:,:,:,:,:,1]
    close(hdf5_file)

    return 
end

@doc raw"""
    read_perambulator(perambulator_file) -> τ_αkβlt

Read the perambulator from the HDF5 file `perambulator_file`.

### Indices
The indices of `τ_αkβlt` are:
- α: sink spinor
- k: sink Laplace mode
- β: source spinor
- l: source Laplace mode
- t: sink time

See also: `read_perambulator!`.
"""
function read_perambulator(perambulator_file)
    # Allocate array and store perambulator in it
    τ_αkβlt = allocate_perambulator()
    read_perambulator!(perambulator_file, τ_αkβlt)

    return τ_αkβlt
end

@doc raw"""
    read_mode_doublet_momenta(mode_doublets_file) -> p_arr

Read the mode\_doublets HDF5 file `mode_doublets_file` and return the momenta `p_arr`.
"""
function read_mode_doublet_momenta(mode_doublets_file)
    # Read momenta from and transpose them such that the p_arr[iₚ, :] is the iₚ'th momentum
    hdf5_file = HDF5.h5open(string(mode_doublets_file), "r")
    p_arr = transpose(read(hdf5_file["axes"]["momenta"]))
    close(hdf5_file)

    return p_arr
end

@doc raw"""
    read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)

Read the mode\_doublets HDF5 file `mode_doublets_file` and store the mode doublets in
`Φ_kltiₚ`. These mode doublets contain no derivatives.

### Indices
The indices of `Φ_kltiₚ` are:
- k:  conjugated Laplace mode
- l:  Laplace mode
- t:  time
- iₚ: momentum

See also: `read_mode_doublets`.
"""
function read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)
    # Check if shape is correct
    N_modes1, N_modes2, Nₜ, _ = size(Φ_kltiₚ)
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("number of modes don't match."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimensions of time axis don't match."))
    end

    # Read mode doublets and set the derivative index to 1 (no derivative)
    hdf5_file = HDF5.h5open(string(mode_doublets_file), "r")
    Φ_tmp_kltiₚ = read(hdf5_file["mode_doublets"])[1,:,:,:,:]
    close(hdf5_file)

    # Permute dimensions to match index convention of perambulator
    permutedims!(Φ_kltiₚ, Φ_tmp_kltiₚ, (2, 1, 3, 4))

    return
end

@doc raw"""
    read_mode_doublets(mode_doublets_file) -> Φ_kltiₚ

Read the mode\_doublets HDF5 file `mode_doublets_file` and return the mode doublets
`Φ_kltiₚ`. These mode doublets contain no derivatives.

### Indices
The indices of `Φ_kltiₚ` are:
- k:  conjugated Laplace mode
- l:  Laplace mode
- t:  time
- iₚ: momentum

See also: `read_mode_doublets!`.
"""
function read_mode_doublets(mode_doublets_file)
    # Allocate array and store mode doublets in it 
    Φ_kltiₚ = allocate_mode_doublets(mode_doublets_file)
    read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)

    return Φ_kltiₚ
end

@doc raw"""
    read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)

Read the sparse\_modes HDF5 file `sparse_modes_file` and store the sparse space positions
at the sink `x_sink_μiₓ` and the source `x_src_μiₓt`, and the sparse modes (eigenvectors
of Laplacian) for the sink `v_sink_μiₓkt` and the source `v_src_μiₓkt` in
`sparse_modes_arrays` = `(x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt)`.

### Indices
The last characters in the variable names describe which indices these arrays carry. The
indices μ, iₓ, t, c and k have the following meaning:
- μ:  spacial direction
- iₓ: lattice position
- t:  time
- c:  color index
- k:  Laplace mode

See also: `read_sparse_modes`.
"""
function read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)
    # Unpack `sparse_modes_arrays`
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Check if shapes are correct
    N_dim1, N_points1 = size(x_sink_μiₓ)
    N_dim2, N_points2, Nₜ = size(x_src_μiₓt)
    N_color1, N_points3, N_modes1, Nₜ = size(v_sink_ciₓkt)
    N_color2, N_points4, N_modes2, Nₜ = size(v_src_ciₓkt)
    if N_dim1 != N_dim2 != 3
        throw(DimensionMismatch("the space is not three dimensional."))
    end
    if N_points1 != N_points2 != N_points3 != N_points4
        throw(DimensionMismatch("the number of points in the spares spaces don't match."))
    end
    if N_color1 != N_color2 != 4
        throw(DimensionMismatch("dimensions of color axis don't match."))
    end
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("number of modes don't match."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimensions of time axis don't macht."))
    end

    hdf5_file = HDF5.h5open(string(sparse_modes_file), "r")
    x_sink_μiₓ[:] = read(hdf5_file["sparse_space_sink"])
    x_src_μiₓt[:] = read(hdf5_file["sparse_space_src"])
    v_sink_ciₓkt[:] = read(hdf5_file["sparse_modes_sink"])
    v_src_ciₓkt[:] = read(hdf5_file["sparse_modes_src"])
    close(hdf5_file)

    return
end

@doc raw"""
    read_sparse_modes(sparse_modes_file)
        -> x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

Read the sparse\_modes HDF5 file `sparse_modes_file` and return the sparse space positions
at the sink `x_sink_μiₓ` and the source `x_src_μiₓt`, and the sparse modes (eigenvectors
of Laplacian) for the sink `v_sink_μiₓkt` and the source `v_src_μiₓkt`.

### Indices
The last characters in the variable names describe which indices these arrays carry. The
indices μ, iₓ, t, c and k have the following meaning:
- μ:  spacial direction
- iₓ: lattice position
- t:  time
- c:  color index
- k:  Laplace mode

See also: `read_sparse_modes!`.
"""
function read_sparse_modes(sparse_modes_file)
    sparse_modes_arrays = allocate_sparse_modes(sparse_modes_file)
    read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)

    return sparse_modes_arrays
end

@doc raw"""
    write_correlator(correlator_file, correlator)

Write `correlator` and its dimension labels to the HDF5 file `correlator_file`.
Additionally, also write the parameter file `parms_toml_string` to it.
"""
function write_correlator(correlator_file, correlator, p=nothing)
    hdf5_file = HDF5.h5open(string(correlator_file), "w")

    # Write correlator with dimension labels
    hdf5_file["Correlator"] = correlator
    HDF5.attributes(hdf5_file["Correlator"])["DIMENSION_LABELS"] = ["t", "source", "cnfg"]

    # Write momentum
    if isnothing(p)
        p = parms.p
    end
    hdf5_file["momentum"] = p

    # Write parameter file
    hdf5_file["parms.toml"] = parms.parms_toml_string
    
    close(hdf5_file)

    return
end
