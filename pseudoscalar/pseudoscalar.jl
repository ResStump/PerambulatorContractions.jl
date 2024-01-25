# %%########################################################################################
# pseudoscalar.jl
#
# Compute the pseudoscalar meson from perambulators and mode doublets
#
# Usage:
#   pseudoscalar.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
############################################################################################

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import BenchmarkTools.@btime

# Add infile manually to arguments
pushfirst!(ARGS, "-i", "pseudoscalar/parms_16x8v1.toml")


# %%###############
# Global Parameters
###################

@doc raw"""
    Parms

Important parameters for the perambulator contractions.
"""
struct Parms
    # String containt in parameter toml file passed to program
    parms_toml_string::String

    # Paths
    dat_dir
    result_dir

    # Configuration numbers and source times
    cnfg_indices::Vector{Int}
    tsrc_arr::Array{Int, 2}

    # Lattice size in time and space and number of modes
    Nₜ::Int
    Nₖ::Vector{Int} # spatial components k = 1, 2, 3
    N_modes::Int

    # Number of configurations and sources
    N_cnfg::Int
    N_src::Int

    # Momentum
    p::Vector{Int}
end

# Instance of Parms
parms = nothing


# Dict with parameters from toml
parms_toml = Dict()





# %%#######
# Functions
###########

# Read and Write Functions
##########################

@doc raw"""
    read_parameters()

Read the parameters stored in the parameter file passed to the program with the flag -i.
Set dictonary 'parms_toml' and Parms instance 'parms'.
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

    # Read parameters from parameter file and store them in the 'parms_toml'
    parms_toml_string = read(parms_file, String)
    merge!(parms_toml, TOML.parse(parms_toml_string))

    # Read source times
    tsrc_list = DelimitedFiles.readdlm(
        parms_toml["Directories and Files"]["tsrc_list"], ' ', Int
    )
    
    # Set paths
    dat_dir = Path(parms_toml["Directories and Files"]["dat_dir"])
    result_dir = Path(parms_toml["Directories and Files"]["result_dir"])

    # Lattice size
    Nₜ = parms_toml["Geometry"]["N_t"]
    Nₖ = parms_toml["Geometry"]["N_k"]

    # Set 'cnfg_indices'
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

    # Store all source times in 'tsrc_arr'
    N_src = parms_toml["Sources"]["N_src"]
    src_separation = parms_toml["Sources"]["src_separation"]
    tsrc_arr = hcat([tsrc_first .+ i_src*src_separation
                     for i_src in 0:N_src-1]...)
    tsrc_arr = mod.(tsrc_arr, Nₜ) # periodically shift values >= Nₜ

    # Extract number of modes from one perambulator file
    perambulator_file = 
        "perambulator_$(parms_toml["Perambulator"]["label_base"])" * "$(tsrc_list[1, 2])_" *
        "$(parms_toml["Run name"]["name"])n$(tsrc_list[1, 1])"
    
    file = HDF5.h5open(string(dat_dir/perambulator_file), "r")
    N_modes = size(file["perambulator"])[4]
    close(file)

    # Momentum
    p = parms_toml["Momentum"]["p"]

    # Store all parameters in 'parms'
    global parms = Parms(parms_toml_string, dat_dir, result_dir, cnfg_indices, tsrc_arr,
                         Nₜ, Nₖ, N_modes, N_cnfg, N_src, p)
end

@doc raw"""
    read_perambulator(perambulator_file) -> τ_αkβlt

Read the perambulator from the HDF5 file 'perambulator_file'.

### Indices
The indices of 'τ_αkβlt' are:
- α: sink spinor
- k: sink Laplace mode
- β: source spinor
- l: source Laplace mode
- t: sink time
"""
function read_perambulator(perambulator_file)
    # Read perambulator and set noise index to 1 (since noise is 'ones')
    hdf5_file = HDF5.h5open(string(perambulator_file), "r")
    τ_αkβlt = read(hdf5_file["perambulator"])[:,:,:,:,:,1]
    close(hdf5_file)

    # Check if shape is correct
    N_color1, N_modes1, N_color2, N_modes2, Nₜ = size(τ_αkβlt)
    if N_color1 != N_color2 != 4
        throw(DimensionMismatch("dimension of spinor index in perambulator is wrong."))
    end
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("wrong number of modes in the perambulator."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimension of time index in perambulator is wrong."))
    end

    return τ_αkβlt
end

@doc raw"""
    read_mode_doublets(mode_doublets_file) -> Φ_kltiₚ

Read the mode\_doublets HDF5 file 'mode\_doublets\_file' and return the mode doublets
'Φ\_kltiₚ'. These mode doublets contain no derivatives.

### Indices
The indices of 'Φ_kltiₚ' are:
- k:  conjugated Laplace mode
- l:  Laplace mode
- t:  time
- iₚ: momentum
"""
function read_mode_doublets(mode_doublets_file)
    # Read mode doublets and set the derivative index to 1 (no derivative)
    hdf5_file = HDF5.h5open(string(mode_doublets_file), "r")
    Φ_kltiₚ = read(hdf5_file["mode_doublets"])[1,:,:,:,:]
    close(hdf5_file)

    # Check if shape is correct
    N_modes1, N_modes2, Nₜ, _ = size(Φ_kltiₚ)
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("wrong number of modes in the mode doublets."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimension of time index in mode doublets is wrong."))
    end

    # Permute dimensions to match index convention of perambulator
    Φ_kltiₚ = permutedims(Φ_kltiₚ, (2, 1, 3, 4))

    return Φ_kltiₚ
end

@doc raw"""
    read_mode_doublet_momenta(mode_doublets_file) -> p_arr

Read the mode\_doublets HDF5 file 'mode\_doublets\_file' and return the momenta 'p\_arr'.
"""
function read_mode_doublet_momenta(mode_doublets_file)
    # Read momenta from and transpose them such that the p_arr[iₚ, :] is the iₚ'th momentum
    hdf5_file = HDF5.h5open(string(mode_doublets_file), "r")
    p_arr = transpose(read(hdf5_file["axes"]["momenta"]))
    close(hdf5_file)

    return p_arr
end

@doc raw"""
    read_sparse_modes(sparse_modes_file)
        -> x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

Read the sparse\_modes HDF5 file 'sparse\_modes\_file' and return the sparse space positions
at the sink 'x\_sink\_μiₓ' and the source 'x\_src\_μiₓt', and the sparse modes (eigenvectors
of Laplacian) for the sink 'v\_sink\_μiₓkt' and the source 'v\_src\_μiₓkt'.

### Indices
The last characters in the variable names describe which indices these arrays carry. The
indices μ, iₓ, t, c and k have the following meaning:
- μ:  spacial direction
- iₓ: lattice position
- t:  time
- c:  color index
- k:  Laplace mode
"""
function read_sparse_modes(sparse_modes_file)
    hdf5_file = HDF5.h5open(string(sparse_modes_file), "r")
    x_sink_μiₓ = read(hdf5_file["sparse_space_sink"])
    x_src_μiₓt = read(hdf5_file["sparse_space_src"])
    v_sink_ciₓkt = read(hdf5_file["sparse_modes_sink"])
    v_src_ciₓkt = read(hdf5_file["sparse_modes_src"])
    close(hdf5_file)

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
        throw(DimensionMismatch("dimension of spinor index in perambulator is wrong."))
    end
    if N_modes1 != N_modes2 != parms.N_modes
        throw(DimensionMismatch("wrong number of modes in the perambulator."))
    end
    if Nₜ != parms.Nₜ
        throw(DimensionMismatch("dimension of time index in perambulator is wrong."))
    end

    return x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt
end

@doc raw"""
    write_correlator(correlator_file, correlator)

Write 'correlator' and its dimension labels to the HDF5 file 'correlator\_file'.
Additionally, also write the parameter file 'parms\_toml\_string' to it.
"""
function write_correlator(correlator_file, correlator)
    hdf5_file = HDF5.h5open(string(correlator_file), "w")

    # Write correlator with dimension labels
    hdf5_file["Correlator"] = correlator
    HDF5.attributes(hdf5_file["Correlator"])["DIMENSION_LABELS"] = ["t", "source", "cnfg"]

    # Write parameter file
    hdf5_file["parms.toml"] = parms.parms_toml_string
    
    close(hdf5_file)

    return
end


# Contraction Functions
#######################

@doc raw"""
    pseudoscalar_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                              Φ_kltiₚ::AbstractArray,  t₀::Integer, iₚ::Integer)

Contract the perambulator 'τ\_αkβlt' and the mode doublet 'Φ\_kltiₚ' to get the
pseudoscalar correlator and store it in 'Cₜ'. The source time 't₀' is used to circularly
shift 'Cₜ' such that the source time is at the origin. The index 'iₚ' sets the momentum
that is used from the mode doublet.
"""
function pseudoscalar_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                                   Φ_kltiₚ::AbstractArray,  t₀::Integer, iₚ::Integer)
    # Index for source time 't₀'
    i_t₀ = mod1(t₀+1, parms.Nₜ)

    # Mode doublet at source time 't₀'
    Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Mode doublet and perambulator at sink time t (index 'iₜ')
        Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :,iₜ]
        
        # Tensor contraction
        TO.@tensoropt begin
            C = Φ_kl_tiₚ[k, k'] * τ_αkβl_t[α, k', β, l'] *
                conj(Φ_kl_t₀iₚ[l, l']) * conj(τ_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, t₀::Integer)

Contract the perambulator 'τ\_αkβlt' to get the pseudoscalar correlator and 
store it in 'Cₜ'. The source time 't₀' is used to circularly shift 'Cₜ' such that the
source time is at the origin.
"""
function pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, 
                                      t₀::Integer)
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulator at sink (index 'iₜ')
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = τ_αkβl_t[α, k, β, l] * conj(τ_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    pseudoscalar_sparse_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                                     sparse_modes_arrays::NTuple{4, AbstractArray},
                                     t₀::Integer, p::Vector)

Contract the perambulator 'τ\_αkβlt' and the sparse Laplace modes stored in
'sparse\_modes\_arrays' to get the pseudoscalar correlator and store it in 'Cₜ'. The source
time 't₀' is used to circularly shift 'Cₜ' such that the source time is at the origin.
The array 'p' is the integer momentum that is used for the momentum projection of the
correlator.
"""
function pseudoscalar_sparse_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                                          sparse_modes_arrays::NTuple{4, AbstractArray},
                                          t₀::Integer, p::Vector)
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time 't₀'
    i_t₀ = mod1(t₀+1, parms.Nₜ)
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Laplace modes at source time 't₀'
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]
    
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulator and Laplace modes at sink time t (index 'iₜ')
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]
        v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

        # Source position for sink time t
        x_src_μiₓ_t = @view x_src_μiₓt[:, :, iₜ]

        # Compute exp(±ipx) and reshape it to match shape of Laplace modes
        exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./parms.Nₖ)'*p)
        exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))
        exp_ipx_src_iₓ = exp.(2π*im * (x_src_μiₓ_t./parms.Nₖ)'*p)
        exp_ipx_src_iₓ = reshape(exp_ipx_src_iₓ, (1, N_points, 1))

        # Tensor contraction
        TO.@tensoropt begin
            C = conj(v_sink_ciₓk_t[a, iₓ', k]) * 
                (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ', k'] *
                τ_αkβl_t[α, k', β, l'] *
                conj(v_src_ciₓk_t₀[b, iₓ, l']) *
                (exp_ipx_src_iₓ .* v_src_ciₓk_t₀)[b, iₓ, l] *
                conj(τ_αkβl_t[α, k, β, l])
        end

        # Normalization
        C *= (prod(parms.Nₖ)/N_points)^2

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end





# %%#########
# Computation
#############

read_parameters()

# Compute Contractions
######################

# File paths
perambulator_file(n_cnfg, i_src) = parms.dat_dir/"perambulator_" *
                                   "$(parms_toml["Perambulator"]["label_base"])$(i_src)_" *
                                   "$(parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = parms.dat_dir/
                             "mode_doublets_$(parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = parms.dat_dir/
                            "sparse_modes_$(parms_toml["Run name"]["name"])n$(n_cnfg)"





# Compare imported mode doublets to those computed from Laplacian eigenmodes
############################################################################

# Read mode doublets
#= Φ_klt_iₚ = read_mode_doublets(mode_doublets_file(1), 2)[:, :, :, iₚ]


# Read eigen modese and the positions
sparse_modes_arrays = read_sparse_modes(sparse_modes_file(1), 2)
x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

x_sink_μiₓ
x_src_μiₓt
v_sink_ciₓkt
v_src_ciₓkt


# Number of lattice points
_, N_points = size(x_sink_μiₓ)


# Create empty mode doublet Array
Φ_klt_iₚ_ = Array{ComplexF64, 3}(undef, parms.N_modes, parms.N_modes, parms.Nₜ)
Φ_klt_iₚ_adj = Array{ComplexF64, 3}(undef, parms.N_modes, parms.N_modes, parms.Nₜ)

for t in 1:parms.Nₜ
    # Laplace modes and mode doublets at time t
    v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, t]
    Φ_kl_tiₚ = @view Φ_klt_iₚ[:, :, t]
    Φ_kl_tiₚ_ = @view Φ_klt_iₚ_[:, :, t]
    Φ_kl_tiₚ_adj = @view Φ_klt_iₚ_adj[:, :, t]

    # Compute exp(±ipx) and reshape it to match shape of v_sink_ciₓk_t
    exp_mipx_sink_iₓ = exp.(-2π*im*(x_sink_μiₓ./parms.Nₖ)'*p)
    exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))

    TO.@tensoropt begin
        Φ_kl_tiₚ_[k, l] = conj(v_sink_ciₓk_t[c, iₓ, k]) * 
                          (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[c, iₓ, l]
    end

    TO.@tensoropt begin
        Φ_kl_tiₚ_adj[k, l] = (v_sink_ciₓk_t[c, iₓ, l]) * 
                          conj((exp_mipx_sink_iₓ) .* v_sink_ciₓk_t)[c, iₓ, k]
    end

    @assert Φ_kl_tiₚ' ≈ Φ_kl_tiₚ_adj
end

Φ_klt_iₚ ≈ Φ_klt_iₚ_
Φ_klt_iₚ ≈ Φ_klt_iₚ_adj =#



# Now follows the actual computation
# %%




# Get momentum index
p_arr = read_mode_doublet_momenta(mode_doublets_file(parms.cnfg_indices[1]))
iₚ = findfirst(p -> p == parms.p, eachrow(p_arr))
if isnothing(iₚ)
    throw(DomainError("the chosen momentum 'p' is not contained in mode doublets."))
end

# Allocate arrays to store the correlator
correlator = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator2 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator3 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)

for (i_cnfg, n_cnfg) in enumerate(parms.cnfg_indices)
    println("Configuration $n_cnfg")

    sparse_modes_arrays = read_sparse_modes(sparse_modes_file(n_cnfg))
    Φ_kltiₚ = read_mode_doublets(mode_doublets_file(n_cnfg))

    for (i_src, t₀) in enumerate(parms.tsrc_arr[i_cnfg, :])
        println("Source: $i_src of $(parms.N_src)")

        τ_αkβlt = read_perambulator(perambulator_file(n_cnfg, t₀))

        Cₜ = @view correlator[:, i_src, n_cnfg]
        Cₜ_2 = @view correlator2[:, i_src, n_cnfg]
        Cₜ_3 = @view correlator3[:, i_src, n_cnfg]
        pseudoscalar_contraction!(Cₜ, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
        pseudoscalar_contraction_p0!(Cₜ_2, τ_αkβlt, t₀)
        pseudoscalar_sparse_contraction!(Cₜ_3, τ_αkβlt, sparse_modes_arrays, t₀, parms.p)
    end
    
    println("Finished configuration $n_cnfg\n")
end


# Store correlator
##################

correlator_file = 
    "$(parms_toml["Run name"]["name"])_" *
    "$(parms.N_modes)modes_pseudoscalar.hdf5"

write_correlator(parms.result_dir/correlator_file, correlator)


# %%

import Plots as Plt
import Statistics as Stats
using LaTeXStrings

corr = vec(Stats.mean(real(correlator), dims=(2, 3)))
corr[corr.<=0] .= NaN
corr2 = vec(Stats.mean(real(correlator2), dims=(2, 3)))
corr2[corr2.<=0] .= NaN
corr3 = vec(Stats.mean(real(correlator3), dims=(2, 3)))
corr3[corr3.<=0] .= NaN

corr_ = Stats.mean(real(correlator), dims=(2))
corr_[corr_.<=0] .= NaN
corr2_ = Stats.mean(real(correlator2), dims=(2))
corr2_[corr2_.<=0] .= NaN

plot = Plt.plot(xlabel=L"t/a", ylabel=L"C(t)", yscale=:log)
Plt.scatter!(1:parms.Nₜ, corr, label="Using mode doublets")
# Plt.scatter!(1:parms.Nₜ, corr2, label="For zero momentum")
Plt.scatter!(1:parms.Nₜ, corr3, label="Position space sampling")
for i in 1:parms.N_cnfg
    #= Plt.plot!(legend=false)
    Plt.scatter!(1:parms.Nₜ, corr_[:, 1, i][corr_[:, 1, i].>0.0], label="Using mode doublets")
    Plt.scatter!(1:parms.Nₜ, corr2_[:, 1, i][corr2_[:, 1, i].>0.0], label="Using full eigenvectors") =#

end
display(plot)

# Plt.savefig(p, "pseudoscalar_p1,0,0_Nsep1.pdf")


# %%