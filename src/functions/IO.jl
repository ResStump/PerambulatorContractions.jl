@doc raw"""
    Parms

Important parameters for the perambulator contractions.
"""
struct Parms
    # String containt in parameter toml file passed to program
    parms_toml_string::String

    # Paths
    perambulator_dir
    perambulator_charm_dir
    mode_doublets_dir
    sparse_modes_dir
    result_dir

    # Configuration numbers and source times
    cnfg_numbers::Vector{Int}
    tsrc_arr::Array{Int, 2}

    # Lattice size in time and space, and number of modes
    Nₜ::Int
    Nₖ::Vector{Int} # spatial components k = 1, 2, 3
    N_modes::Int

    # Number of configurations and sources
    N_cnfg::Int
    N_src::Int

    # Momentum indices and Momenta
    iₚ_arr::Vector{Int}
    p_arr::Vector{Vector{Int}}

    # Number of ranks that symultaneously work on one configuration
    N_ranks_per_cnfg::Int
end

parms = nothing
parms_toml = nothing

@doc raw"""
    read_parameters()

Read the parameters stored in the parameter file passed to the program with the flag -i and
return the dictonary `parms_toml` and the Parms instance `parms`.
"""
function read_parameters()
    # Parse arguments
    s = AP.ArgParseSettings()
    AP.@add_arg_table s begin
        "-i"
            help = "Input file."
            arg_type = String
            required = true
        "--nranks-per-cnfg"
            help = "Number of ranks per configuration."
            arg_type = Int
            default = 1
            required = false
    end
    args = AP.parse_args(s)

    # Input file path
    parms_file = args["i"]

    # Number of ranks per configuration
    N_ranks_per_cnfg = args["nranks-per-cnfg"]

    # Read parameters from parameter file and store them in the `parms_toml`
    parms_toml_string = read(parms_file, String)
    global parms_toml = TOML.parse(parms_toml_string)

    # Add information about program to `parms_toml` (as string)
    parms_toml["Program Information"] =
        "Date = $(Dates.now())\n"*
        "Julia version = $VERSION\n"*
        "$(@__MODULE__) version = $(pkgversion(@__MODULE__))\n"*
        "Program file = $PROGRAM_FILE\n"

    # Read source times
    tsrc_list = DelimitedFiles.readdlm(
        parms_toml["Directories and Files"]["tsrc_list"], ' ', Int
    )
    
    # Set paths
    perambulator_dir = Path(parms_toml["Directories and Files"]["perambulator_dir"])
    perambulator_charm_dir =
        Path(parms_toml["Directories and Files"]["perambulator_charm_dir"])
    mode_doublets_dir = Path(parms_toml["Directories and Files"]["mode_doublets_dir"])
    sparse_modes_dir = Path(parms_toml["Directories and Files"]["sparse_modes_dir"])
    result_dir = Path(parms_toml["Directories and Files"]["result_dir"])

    # Lattice size
    Nₜ = parms_toml["Geometry"]["N_t"]
    Nₖ = parms_toml["Geometry"]["N_k"]

    # Set `cnfg_numbers`
    first_cnfg = parms_toml["Configurations"]["first"]
    step_cnfg = parms_toml["Configurations"]["step"]
    last_cnfg = parms_toml["Configurations"]["last"]
    N_cnfg = (last_cnfg - first_cnfg) ÷ step_cnfg + 1
    cnfg_numbers = Array(first_cnfg:step_cnfg:last_cnfg)

    # Find first source times for specified configurations
    tsrc_first = Array{Int, 1}(undef, N_cnfg)
    for (i_cnfg, n_cnfg) in enumerate(cnfg_numbers)
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
        "$(parms_toml["Perambulator"]["label_light"])" * "$(tsrc_list[1, 2])_" *
        "$(parms_toml["Run name"]["name"])n$(tsrc_list[1, 1])"
    
    file = HDF5.h5open(string(perambulator_dir/perambulator_file), "r")
    N_modes = size(file["perambulator"])[4]
    close(file)

    # Path to a mode doublets file
    mode_doublets_file = mode_doublets_dir/
        "mode_doublets_$(parms_toml["Run name"]["name"])n$(tsrc_list[1, 1])"

    # Momenta and the corresponding indices
    p_arr = read_mode_doublet_momenta(mode_doublets_file)
    iₚ_arr = collect(1:length(p_arr))

    # Store all parameters
    global parms = Parms(parms_toml_string, perambulator_dir, perambulator_charm_dir,
                         mode_doublets_dir, sparse_modes_dir, result_dir, cnfg_numbers,
                         tsrc_arr, Nₜ, Nₖ, N_modes, N_cnfg, N_src, iₚ_arr, p_arr,
                         N_ranks_per_cnfg)
    
    return
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
    file = HDF5.h5open(string(perambulator_file), "r")
    τ_αkβlt[:] = file["perambulator"][:,:,:,:,:,1]
    close(file)

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

Read the mode doublets HDF5 file `mode_doublets_file` and return the momenta `p_arr`.
"""
function read_mode_doublet_momenta(mode_doublets_file)
    # Read momenta from and transpose them such that the p_arr[iₚ, :] is the iₚ'th momentum
    file = HDF5.h5open(string(mode_doublets_file), "r")
    p_μiₚ = read(file["axes"]["momenta"])
    close(file)

    # Convert to vector of vectors
    p_arr = [collect(p) for p in eachcol(p_μiₚ)]

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
    file = HDF5.h5open(string(mode_doublets_file), "r")
    Φ_tmp_kltiₚ = file["mode_doublets"][1,:,:,:,:]
    close(file)

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
at the sink `x_sink_μiₓt` and the source `x_src_μiₓt`, and the sparse modes (eigenvectors
of Laplacian) for the sink `v_sink_μiₓkt` and the source `v_src_μiₓkt` in
`sparse_modes_arrays` = `(x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt)`.

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
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Check if shapes are correct
    N_dim1, N_points1, Nₜ1 = size(x_sink_μiₓt)
    N_dim2, N_points2, Nₜ2 = size(x_src_μiₓt)
    N_color1, N_points3, N_modes1, Nₜ3 = size(v_sink_ciₓkt)
    N_color2, N_points4, N_modes2, Nₜ4 = size(v_src_ciₓkt)
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
    if Nₜ1 != Nₜ2 != Nₜ3 != Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimensions of time axis don't match."))
    end

    file = HDF5.h5open(string(sparse_modes_file), "r")
    x_sink_μiₓt[:] = read(file["sparse_space_sink"])
    x_src_μiₓt[:] = read(file["sparse_space_src"])
    v_sink_ciₓkt[:] = read(file["sparse_modes_sink"])
    v_src_ciₓkt[:] = read(file["sparse_modes_src"])
    close(file)

    return
end

@doc raw"""
    read_sparse_modes(sparse_modes_file)
        -> x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

Read the sparse\_modes HDF5 file `sparse_modes_file` and return the sparse space positions
at the sink `x_sink_μiₓt` and the source `x_src_μiₓt`, and the sparse modes (eigenvectors
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
    write_sparse_modes(sparse_modes_file, sparse_modes_arrays)

Write the sparse modes in `sparse_modes_arrays` to the HDF5 file `sparse_modes_file`.

The `sparse_modes_arrays` must contain the sparse space positions at the sink `x_sink_μiₓt`
and the source `x_src_μiₓt`, and the sparse modes (eigenvectors of Laplacian) for the sink
`v_sink_μiₓkt` and the source `v_src_μiₓkt`.

See the documentation of the `read_sparse_modes` function for further information.
"""
function write_sparse_modes(sparse_modes_file, sparse_modes_arrays)
    # Unpack `sparse_modes_arrays`
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    file = HDF5.h5open(string(sparse_modes_file), "w")

    # Write sparse spaces
    file["sparse_space_sink"] = x_sink_μiₓt
    file["sparse_space_src"] = x_src_μiₓt

    # Write sparse modes
    file["sparse_modes_sink"] = v_sink_ciₓkt
    file["sparse_modes_src"] = v_src_ciₓkt

    # Set attributes
    HDF5.attrs(file["sparse_space_sink"])["DIMENSION_LABELS"] =
        ["sink t", "position", "component"]
    HDF5.attrs(file["sparse_space_src"])["DIMENSION_LABELS"] =
        ["source t", "position", "component"]
    HDF5.attrs(file["sparse_modes_sink"])["DIMENSION_LABELS"] =
        ["t", "Laplace mode", "position", "color"]
    HDF5.attrs(file["sparse_modes_src"])["DIMENSION_LABELS"] =
        ["t", "Laplace mode", "position", "color"]

    close(file)

    return
end