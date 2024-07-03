# %%########################################################################################
# DD_nolocal.jl
#
# Compute nonlocal DD correlators from perambulators and mode doublets.
#
# Usage:
#   DD_nolocal.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters.
#
############################################################################################

import MKL
import LinearAlgebra as LA
import MPI
import HDF5
import DelimitedFiles as DF
import FilePathsBase: /, Path
import BenchmarkTools.@btime
#= import PerambulatorContractions as PC =#

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)
N_ranks = MPI.Comm_size(comm)

if myrank != 0
    redirect_stdout(devnull)
end


# %%###############
# Global Parameters
###################

# Set global parameters
PC.read_parameters()

# Array of (monomial of) γ-matrices
Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ = length(Γ_arr)

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(myrank).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end


# %%############################
# Generate Momentum Combinations
################################

# Array of square of total angular momentas
Ptot_sq_arr = PC.parms_toml["Momenta"]["Ptot_sq"]

# Maximal sum of squares of the momentum pairs that are used
p_sq_sum_max_arr = PC.parms_toml["Momenta"]["p_sq_sum_max"]

# Compute all combinations of the momentum indeces at sink and source
Iₚ_arr = []
for (Ptot_sq, p_sq_sum_max) in zip(Ptot_sq_arr, p_sq_sum_max_arr)
    # Pairs of momenta
    Iₚ_pair_arr = PC.generate_momentum_pairs(Ptot_sq, p_sq_sum_max)

    # All combinations of sink and source momentum pairs
    for (Iₚ_pair_1, Iₚ_pair_2) in Iterators.product(Iₚ_pair_arr, Iₚ_pair_arr)
        push!(Iₚ_arr, [Iₚ_pair_1..., Iₚ_pair_2...])
    end
end


# %%############
# File Functions
################

# File paths
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/
    "$(PC.parms_toml["Perambulator"]["label_light"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
perambulator_charm_file(n_cnfg, i_src) = PC.parms.perambulator_charm_dir/
    "$(PC.parms_toml["Perambulator"]["label_charm"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"

function write_correlator(n_cnfg, t₀)
    file_path = PC.parms.result_dir/"correlators_DD_nonlocal_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    hdf5_file = HDF5.h5open(string(file_path), "w")

    # Loop over all momentum index combinations
    for (i_p, Iₚ) in enumerate(Iₚ_arr)
        # Get momenta
        p₁, p₂, p₃, p₄ = PC.parms.p_arr[Iₚ]
        @assert p₁ + p₂ == p₃ + p₄

        # Paths to groups in hdf5 file
        Ptot_str = join(p₁ + p₂, ",")
        p₁_str, p₃_str = join.([p₁, p₃], ",")
        group_ūcd̄c_c̄uc̄d = "Correlators/Ptot$(Ptot_str)/psink1_$(p₁_str)/" *
            "psrc1_$(p₃_str)/ubar_c_dbar_c-cbar_u_cbar_d"
        group_ūcd̄c_c̄dc̄u = "Correlators/Ptot$(Ptot_str)/psink1_$(p₁_str)/" *
            "psrc1_$(p₃_str)/ubar_c_dbar_c-cbar_d_cbar_u"

        # Write correlators with dimension labels
        hdf5_file[group_ūcd̄c_c̄uc̄d] = 
            C_ūcd̄c_c̄uc̄d_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
        HDF5.attrs(hdf5_file[group_ūcd̄c_c̄uc̄d])["DIMENSION_LABELS"] = labels
        hdf5_file[group_ūcd̄c_c̄dc̄u] = 
            C_ūcd̄c_c̄dc̄u_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
        HDF5.attrs(hdf5_file[group_ūcd̄c_c̄dc̄u])["DIMENSION_LABELS"] = labels
    end

    # Write parameter file and program information
    hdf5_file["parms.toml"] = PC.parms.parms_toml_string
    hdf5_file["Program Information"] = PC.parms_toml["Program Information"]

    close(hdf5_file)

    return
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulators and mode doublets arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))

# Correlator and its labels
correlator_size = (PC.parms.Nₜ, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(Iₚ_arr))
C_ūcd̄c_c̄uc̄d_tnmn̄m̄Iₚ = Array{ComplexF64}(undef, correlator_size)
C_ūcd̄c_c̄dc̄u_tnmn̄m̄Iₚ = Array{ComplexF64}(undef, correlator_size)
# Reversed order in Julia
labels = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "t"]


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    # Loop over all momentum index combinations
    for (i_p, Iₚ) in enumerate(Iₚ_arr)
        println("    Momenta: $(PC.parms.p_arr[Iₚ])")
        
        @time "      DD nolocal contractons" begin
            # Contraction for correlator of form <(ūc d̄c)(c̄u c̄d)>
            C_ūcd̄c_c̄uc̄d_tnmn̄m̄_Iₚ = @view C_ūcd̄c_c̄uc̄d_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
            PC.DD_nonlocal_contractons!(
                C_ūcd̄c_c̄uc̄d_tnmn̄m̄_Iₚ, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ_arr, t₀, Iₚ
            )
            
            # Contraction for correlator of form <(ūc d̄c)(c̄d c̄u)> (u<->d at source)
            C_ūcd̄c_c̄dc̄u_tnmn̄m̄_Iₚ = @view C_ūcd̄c_c̄dc̄u_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
            PC.DD_nonlocal_contractons!(
                C_ūcd̄c_c̄dc̄u_tnmn̄m̄_Iₚ, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ_arr, t₀, Iₚ,
                swap_ud=true
            )
        end
        println()
    end
end


function main()
    # Loop over all configurations
    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_indices)
        # Skip the cnfgs this rank doesn't have to compute
        if !PC.is_my_cnfg(i_cnfg)
            continue
        end
        if continuation_run && (n_cnfg in finished_cnfgs)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            @time "  Read mode doublets" begin
                PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                @time "    Read perambulators" begin
                    PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                    PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                          τ_charm_αkβlt)
                end
                println()

                compute_contractions!(t₀)
                
                # Write Correlator
                @time "    Write correlator" begin
                    write_correlator(n_cnfg, t₀)
                end
                println()
            end

            # Update finished_cnfgs
            push!(finished_cnfgs, n_cnfg)
            DF.writedlm(string(finished_cnfgs_file), finished_cnfgs, '\n')
        end
        println("\n")

        # Run garbage collector
        GC.gc()
    end

    # Wait until all ranks finished
    MPI.Barrier(comm)

    # Remove finished_cnfgs file
    rm(finished_cnfgs_file, force=true)
end

main()

# %%
