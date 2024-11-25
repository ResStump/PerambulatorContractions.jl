# %%########################################################################################
# DD_nolocal_distributed.jl
#
# Compute nonlocal DD correlators from perambulators and mode doublets where the
# contractions are done in parallel on each MPI rank using Distributed.jl.
#
# Usage:
#   DD_nolocal_distributed.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters.
#
############################################################################################

import Distributed as D

D.@everywhere begin
    import MKL
    import LinearAlgebra as LA
    import MPI
    import HDF5
    import DelimitedFiles as DF
    import FilePathsBase: /, Path
    import BenchmarkTools.@btime
    import PerambulatorContractions as PC
end

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

D.@everywhere begin
    # Broadcast global parameters
    PC.parms = $(PC.parms)
    PC.parms_toml = $(PC.parms_toml)

    # Array of (monomial of) γ-matrices and their labels
    Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
    Nᵧ = length(Γ_arr)
    Γ_DD_labels = ["gamma_5", "gamma_1", "gamma_2", "gamma_3", "-i1"]
end

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
    # 4-tuple of momenta
    Iₚ_arr_ = PC.generate_momentum_4tuples(Ptot_sq, p_sq_sum_max)
    append!(Iₚ_arr, Iₚ_arr_)
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
    file = HDF5.h5open(string(file_path), "w")

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
        file[group_ūcd̄c_c̄uc̄d] = 
            C_ūcd̄c_c̄uc̄d_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
        HDF5.attrs(file[group_ūcd̄c_c̄uc̄d])["DIMENSION_LABELS"] = labels
        file[group_ūcd̄c_c̄dc̄u] = 
            C_ūcd̄c_c̄dc̄u_tnmn̄m̄Iₚ[:, :, :, :, :, i_p]
        HDF5.attrs(file[group_ūcd̄c_c̄dc̄u])["DIMENSION_LABELS"] = labels
    end

    # Write spin structure
    file["Spin Structure/Gamma_DD_1"] = Γ_DD_labels
    file["Spin Structure/Gamma_DD_2"] = Γ_DD_labels

    # Write parameter file and program information
    file["parms.toml"] = PC.parms.parms_toml_string
    file["Program Information"] = PC.parms_toml["Program Information"]

    close(file)

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
    @time "      DD nolocal contractons" begin
        # Index for source time `t₀`
        i_t₀ = t₀+1

        # Convert arrays to vectors of arrays in the time axis
        τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)
        τ_arr = eachslice(τ_αkβlt, dims=5)
        Φ_arr = eachslice(Φ_kltiₚ, dims=3)

        # Select source time `t₀`
        Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]

        # Function to compute contractions
        contractions = (τ_charm, τ, Φ_t) -> begin
            C1_arr = []
            C2_arr = []

            # Loop over all momentum index combinations
            for Iₚ in Iₚ_arr
                C1 = PC.DD_nonlocal_contractons(τ_charm, τ, Φ_t, Φ_kliₚ_t₀, Γ_arr, Iₚ)
                C2 = PC.DD_nonlocal_contractons(τ_charm, τ, Φ_t, Φ_kliₚ_t₀, Γ_arr, Iₚ,
                                                swap_ud=true)
                push!(C1_arr, C1)
                push!(C2_arr, C2)
            end

            # Return as contiguous arrays
            return stack(C1_arr), stack(C2_arr)
        end
          
        # Distribute workload and fetch result
        corr_arr = D.pmap(contractions, τ_charm_arr, τ_arr, Φ_arr)

        # Store correlator entries
        for iₜ in 1:PC.parms.Nₜ
            # Time index for storing correlator entry
            i_Δt = mod1(iₜ-t₀, PC.parms.Nₜ)

            C_ūcd̄c_c̄uc̄d_tnmn̄m̄Iₚ[i_Δt, :, :, :, :, :] = corr_arr[iₜ][1]
            C_ūcd̄c_c̄dc̄u_tnmn̄m̄Iₚ[i_Δt, :, :, :, :, :] = corr_arr[iₜ][2]
        end
    end

    println()
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
        D.@everywhere GC.gc()
    end

    # Wait until all ranks finished
    MPI.Barrier(comm)

    # Remove finished_cnfgs file
    rm(finished_cnfgs_file, force=true)
end

main()

# %%
