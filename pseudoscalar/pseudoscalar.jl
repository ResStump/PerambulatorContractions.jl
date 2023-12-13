# %%############################################################################
# pseudoscalar.jl
#
# Compute the pseudoscalar meson from perambulators and mode doublets
#
# Usage:
#   pseudoscalar.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
################################################################################

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path

# Add infile manually to arguments
push!(ARGS, "-i", "pseudoscalar/parms_B450r000.toml")


# %%###############
# Global Parameters
###################

struct Parms
    # String containt in parameter toml file passed to program
    parms_toml_string::String

    # Paths
    dat_dir
    result_dir

    # Configuration numbers and source times
    cnfg_indices::Vector{Int}
    tsrc_arr::Array{Int, 2}

    # Time extent of lattice and number of modes
    Nₜ::Int
    N_modes::Int

    # Number of configurations and sources
    N_cnfg::Int
    N_src::Int
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

"""
    read_parameters()

Read the parameters stored in parameter file passed to the program with the
flag -i. Set dictonary 'parms_toml' and Parms instance 'parms'.
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

    # Read parameters from parameter file and store them in the Dict 'parms'
    parms_string = read(parms_file, String)
    merge!(parms_toml, TOML.parse(parms_string))

    # Read source times
    tsrc_list = DelimitedFiles.readdlm(
        parms_toml["Directories and Files"]["tsrc_list"], ' ', Int
    )
    
    # Set paths
    dat_dir = Path(parms_toml["Directories and Files"]["dat_dir"])
    result_dir = Path(parms_toml["Directories and Files"]["result_dir"])

    # Extract time extent of lattice and number of modes from perambulator file
    perambulator_file = 
        "perambulator_$(parms_toml["Perambulator"]["label_base"])" *
        "$(tsrc_list[1, 2])_" *
        "$(parms_toml["Run name"]["name"])n$(tsrc_list[1, 1])"
    
    file = HDF5.h5open(string(dat_dir/perambulator_file), "r")
    N_modes, Nₜ = size(file["perambulator"])[4:5]
    close(file)

    # Set cnfg_indices
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

    # Store all source times in tsrc_arr
    N_src = parms_toml["Sources"]["N_src"]
    src_separation = parms_toml["Sources"]["src_separation"]
    tsrc_arr = hcat([tsrc_first .+ i_src*src_separation
                     for i_src in 0:N_src-1]...)
    tsrc_arr = mod.(tsrc_arr, Nₜ) # avoid values >= Nₜ

    global parms =  Parms(parms_string, dat_dir, result_dir, cnfg_indices,
                          tsrc_arr, Nₜ, N_modes, N_cnfg, N_src)
end

"""
    read_perambulator(perambulator_file) -> τ_αkβlt

Read the perambulator from the HDF5 file 'perambulator_file'.

The indices of 'τ_αkβlt' are: \\
    t: sink time \\
    k: sink laplace mode \\
    l: source laplace mode \\
    α: sink spinor \\
    β: source spinor
"""
function read_perambulator(perambulator_file)
    hdf5_file = HDF5.h5open(string(perambulator_file), "r")

    τ_αkβlt = read(hdf5_file["perambulator"])[:,:,:,:,:,1]

    close(hdf5_file)

    return τ_αkβlt
end

"""
    read_mode_doublets(mode_doublets_file) -> Φ_kltiₚ, p_arr

Read the mode_doublets HDF5 file 'mode_doublets_file' and return the
mode_doublets 'Φ_kltiₚ' and the momentas 'p_arr'

The indices of 'Φ_kltiₚ' are: \\
    iₚ: momentum \\
    t:  sink time \\
    k:  sink laplace mode \\
    l:  source laplace mode
"""
function read_mode_doublets(mode_doublets_file)
    hdf5_file = HDF5.h5open(string(mode_doublets_file), "r")
    
    p_arr = transpose(read(hdf5_file["axes"]["momenta"]))
    Φ_klt = read(hdf5_file["mode_doublets"])[1,:,:,:,:]
    
    close(hdf5_file)

    return Φ_klt, p_arr
end

"""
    write_correlator(correlator_file, correlator)

Write 'correlator' and its dimension labels to the HDF5 file 'correlator_file'.
Additionally, also write the parameter file 'parms_toml_string' to it.
"""
function write_correlator(correlator_file, correlator)
    hdf5_file = HDF5.h5open(string(correlator_file), "w")

    # Write correlator with dimension labels
    hdf5_file["Correlator"] = correlator
    HDF5.attributes(hdf5_file["Correlator"])["DIMENSION_LABELS"] = 
        ["t", "source", "cnfg"]

    # Write parameter file
    hdf5_file["parms.toml"] = parms.parms_toml_string
    
    close(hdf5_file)

    return
end


# Contraction Functions
#######################

"""
    pseudoscalar_contraction!(Cₜ, t₀, iₚ, perambulator_file, mode_doublets_file)

Read the perambulator and mode_doublets in 'perambulator_file' and
'mode_doublets_file', and contract them to get the pseudoscalar correlator and
store it in 'Cₜ' The index 'iₚ' sets the momentum.
The resulting correlator is shifted such that the source time 't₀' is the first
entry.
"""
function pseudoscalar_contraction!(Cₜ::Vector{Complex}, t₀, iₚ,
                                   perambulator_file, mode_doublets_file)
    # read perambulator and mode doublets
    τ_αkβlt = read_perambulator(perambulator_file)
    Φ_kltiₚ = read_mode_doublets(mode_doublets_file)[1]

    Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, 1, iₚ]

    for t in 1:parms.Nₜ
        Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, t, iₚ]
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, t]
        
        TO.@tensoropt begin
            C = Φ_kl_tiₚ[k, k'] * τ_αkβl_t[α, k', β, l'] *
                conj(Φ_kl_t₀iₚ[l, l']) * conj(τ_αkβl_t[α, k, β, l])
        end

        # Correct that t₀≠0
        Cₜ[mod1(t-t₀, parms.Nₜ)] = C
    end

    return
end

"""
    pseudoscalar_contraction_p0!(perambulator_file, t₀)

Read the perambulator in 'perambulator_file' and contract it to get the
pseudoscalar correlator for momentum p=0. Then store it in 'Cₜ'.
The resulting correlator is shifted such that the source time 't₀' is the first
entry.
"""
function pseudoscalar_contraction_p0!(Cₜ, t₀, perambulator_file)
    # read perambulator
    τ_αkβlt = read_perambulator(perambulator_file)

    for t in 1:parms.Nₜ
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, t]

        TO.@tensoropt begin
            C = τ_αkβl_t[α, k, β, l] * conj(τ_αkβl_t[α, k, β, l])
        end

        # Correct that t₀≠0
        Cₜ[mod1(t-t₀, parms.Nₜ)] = C
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
perambulator_file(n_cnfg, i_src) = 
    parms.dat_dir/"perambulator_" *
    "$(parms_toml["Perambulator"]["label_base"])$(i_src)_" *
    "$(parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = parms.dat_dir/
    "mode_doublets_$(parms_toml["Run name"]["name"])n$(n_cnfg)"


# Momentum index
iₚ = parms_toml["Mode doublets"]["i_p"]
correlator = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)

for (i_cnfg, n_cnfg) in enumerate(parms.cnfg_indices)
    println("Configuration $n_cnfg")

    for (i_src, t₀) in enumerate(parms.tsrc_arr[i_cnfg, :])
        println("Source: $i_src of $(parms.N_src)")

        Cₜ = @view correlator[:, i_src, n_cnfg]
        #= pseudoscalar_contraction!(Cₜ, t₀, iₚ,
                                 perambulator_file(n_cnfg, t₀),
                                 mode_doublets_file(n_cnfg)) =#
        pseudoscalar_contraction_p0!(Cₜ, t₀, perambulator_file(n_cnfg, t₀))
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