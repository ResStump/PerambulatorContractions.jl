# %%########################################################################################
# meson_connected.jl
#
# Compute meson correlators from perambulators, mode doublets and sparse modes
#
# Usage:
#   meson_connected.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
############################################################################################

import MKL
import LinearAlgebra as LA
import MPI
import HDF5
import DelimitedFiles as DF
import FilePathsBase: /, Path
import BenchmarkTools.@btime
import PerambulatorContractions as PC

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)
N_ranks = MPI.Comm_size(comm)

if myrank != 0
    redirect_stdout(devnull)
end


# %%#############################
# Global Parameters and Functions
#################################

# Set global parameters
PC.read_parameters()

run_name = PC.parms_toml["Run name"]["name"]

# File paths
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/
    "$(PC.parms_toml["Perambulator"]["label_light"])$(i_src)_$(run_name)n$(n_cnfg)"
perambulator_charm_file(n_cnfg, i_src) = PC.parms.perambulator_charm_dir/
    "$(PC.parms_toml["Perambulator"]["label_charm"])$(i_src)_$(run_name)n$(n_cnfg)"
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(run_name)n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(run_name)n$(n_cnfg)"


# Correlator files
function correlator_file(name, p; tmp=false)
    p_str = "p"*join(p, ",")

    file = PC.parms.result_dir/"$(run_name)_$(PC.parms.N_modes)modes_$(name)_"
    file *= p_str
    if tmp
        file *= "_tmp$(myrank).hdf5"
    else
        file *= ".hdf5"
    end
    
    return file
end

correlator1_file_arr = [correlator_file("Dstar_i1", p) for p in PC.parms.p_arr]
correlator2_file_arr = [correlator_file("Dstar_i2", p) for p in PC.parms.p_arr]
correlator3_file_arr = [correlator_file("Dstar_i3", p) for p in PC.parms.p_arr]

correlator1_file_tmp_arr = [correlator_file("Dstar_i1", p, tmp=true)
                            for p in PC.parms.p_arr]
correlator2_file_tmp_arr = [correlator_file("Dstar_i2", p, tmp=true)
                            for p in PC.parms.p_arr]
correlator3_file_tmp_arr = [correlator_file("Dstar_i3", p, tmp=true)
                            for p in PC.parms.p_arr]

# Use sparse modes?
if PC.parms_toml["Correlator"]["method"] == "sparse"
    method = "sparse"
elseif PC.parms_toml["Correlator"]["method"] == "full"
    method = "full"
else
    throw(ArgumentError("correlator method is not valid! Choose \"full\" or \"sparse\""))
end

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(myrank).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulator
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()

# Mode doublets arrays
if method == "full"
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))
end

# Sparse mode arrays
if method == "sparse"
    sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))
end

# Correlator for each momentum
if continuation_run
    correlators1 = PC.read_correlator.(correlator1_file_tmp_arr)
    correlators2 = PC.read_correlator.(correlator2_file_tmp_arr)
    correlators3 = PC.read_correlator.(correlator3_file_tmp_arr)
else
    correlator_size = PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg
    correlators1 = [Array{ComplexF64}(undef, correlator_size) for p in PC.parms.p_arr]
    correlators2 = [Array{ComplexF64}(undef, correlator_size) for p in PC.parms.p_arr]
    correlators3 = [Array{ComplexF64}(undef, correlator_size) for p in PC.parms.p_arr]
end

# Get momentum indices from mode doublets
iₚ_arr = PC.momentum_indices_mode_doublets(mode_doublets_file(n_cnfg))



function compute_contractions!(i_cnfg, i_src, t₀)
    # Loop over all momenta
    for (i_p, p) in enumerate(PC.parms.p_arr)
        println("    Momentum p = $p")
        iₚ = iₚ_arr[i_p]

        # Matrices in interpolators
        Γ₁, Γbar₁ = PC.γ[1], -PC.γ[1]
        Γ₂, Γbar₂ = PC.γ[2], -PC.γ[2]
        Γ₃, Γbar₃ = PC.γ[3], -PC.γ[3]

        # Compute correlator entries
        Cₜ_1 = @view correlators1[i_p][:, i_src, i_cnfg]
        Cₜ_2 = @view correlators2[i_p][:, i_src, i_cnfg]
        Cₜ_3 = @view correlators3[i_p][:, i_src, i_cnfg]
        if method == "full"
            @time "      meson_connected_contraction! (3x)" begin
                PC.meson_connected_contraction!(
                    Cₜ_1, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ₁, Γbar₁, t₀, iₚ
                )
                PC.meson_connected_contraction!(
                    Cₜ_2, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ₂, Γbar₂, t₀, iₚ
                )
                PC.meson_connected_contraction!(
                    Cₜ_3, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ₃, Γbar₃, t₀, iₚ
                )
            end
        else
            @time "      meson_connected_sparse_contraction! (3x)" begin
                PC.meson_connected_sparse_contraction!(
                    Cₜ_1, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ₁, Γbar₁, t₀, p
                )
                PC.meson_connected_sparse_contraction!(
                    Cₜ_2, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ₂, Γbar₂, t₀, p
                )
                PC.meson_connected_sparse_contraction!(
                    Cₜ_3, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ₃, Γbar₃, t₀, p
                )
            end
        end
        println()
    end
end


function main()
    # Computation
    #############

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
            if method == "full"
                @time "  Read mode doublets" begin
                    PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
                end
            else
                @time "  Read sparse modes " begin
                    PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
                end
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

                compute_contractions!(i_cnfg, i_src, t₀)
                
                println()
            end

            # Temporary store correlators and update finished_cnfgs
            @time "  Write tmp Files" begin
                PC.write_correlator.(correlator1_file_tmp_arr, correlators1, PC.parms.p_arr)
                PC.write_correlator.(correlator2_file_tmp_arr, correlators2, PC.parms.p_arr)
                PC.write_correlator.(correlator3_file_tmp_arr, correlators3, PC.parms.p_arr)

                push!(finished_cnfgs, n_cnfg)
                DF.writedlm(string(finished_cnfgs_file), finished_cnfgs, '\n')
            end
            println()
        end
        println("\n")

        # Run garbage collector
        GC.gc()
    end

    # Broadcast correlators to all ranks
    @time "Broadcast correlators" begin
        PC.broadcast_correlators!.(correlators1)
        PC.broadcast_correlators!.(correlators2)
        PC.broadcast_correlators!.(correlators3)
    end

    # Store correlators and remove tmp correlators
    ##############################################

    if myrank == 0
        @time "Write correlators" begin
            PC.write_correlator.(correlator1_file_arr, correlators1, PC.parms.p_arr)
            PC.write_correlator.(correlator2_file_arr, correlators2, PC.parms.p_arr)
            PC.write_correlator.(correlator3_file_arr, correlators3, PC.parms.p_arr)
        end
    end

    @time "Remove tmp correlators" begin
        rm.(correlator1_file_tmp_arr, force=true)
        rm.(correlator2_file_tmp_arr, force=true)
        rm.(correlator3_file_tmp_arr, force=true)
        rm(finished_cnfgs_file, force=true)
    end

end

main()


# %%