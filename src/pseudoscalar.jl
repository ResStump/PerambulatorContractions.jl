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
import MPI
import HDF5
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


# File paths
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/
    "$(PC.parms_toml["Perambulator"]["label_light"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
perambulator_charm_file(n_cnfg, i_src) = PC.parms.perambulator_charm_dir/
    "$(PC.parms_toml["Perambulator"]["label_charm"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"


# Correlator files
function correlator_file(name, p; tmp=false)
    run_name = PC.parms_toml["Run name"]["name"]
    p_str = "p"*join(p, ",")

    file = PC.parms.result_dir/"$(run_name)_$(PC.parms.N_modes)modes_$(name)_"
    file *= p_str
    if tmp
        file *= "_tmp.hdf5"
    else
        file *= ".hdf5"
    end
    
    return file
end

correlator1_file_arr = [correlator_file("pseudoscalar", p) for p in PC.parms.p_arr]
correlator2_file_arr = [correlator_file("pseudoscalar_sparse", p) for p in PC.parms.p_arr]

correlator1_file_tmp_arr = [correlator_file("pseudoscalar", p, tmp=true)
                            for p in PC.parms.p_arr]
correlator2_file_tmp_arr = [correlator_file("pseudoscalar_sparse", p, tmp=true)
                            for p in PC.parms.p_arr]


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulator and mode doublets arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))

# Sparse mode arrays
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

# Correlator for each momentum
correlator_size = PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg
correlators1 = [Array{ComplexF64}(undef, correlator_size) for p in PC.parms.p_arr]
correlators2 = [Array{ComplexF64}(undef, correlator_size) for p in PC.parms.p_arr]




function main()
    # Get momentum indices from mode doublets
    iₚ_arr = PC.momentum_indices_mode_doublets(mode_doublets_file(n_cnfg))

    # Computation
    #############

    # Loop over all configurations
    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_indices)
        # Skip the cnfgs this rank doesn't have to compute
        if !PC.is_my_cnfg(i_cnfg)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            @time "  Read sparse modes " begin
                PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
            end
            @time "  Read mode doublets" begin
                PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                # Loop over all momenta
                for (i_p, p) in enumerate(PC.parms.p_arr)
                    println("    Momentum p = $p")
                    iₚ = iₚ_arr[i_p]

                    @time "      Read perambulators" begin
                        PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                        PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                              τ_charm_αkβlt)
                    end
                    println()

                    # Compute correlator entries
                    Cₜ = @view correlators1[i_p][:, i_src, i_cnfg]
                    Cₜ_2 = @view correlators2[i_p][:, i_src, i_cnfg]
                    @time "      pseudoscalar_contraction!       " begin
                        PC.pseudoscalar_contraction!(Cₜ, τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ,
                                                     t₀, iₚ)
                    end
                    @time "      pseudoscalar_sparse_contraction!" begin
                        PC.pseudoscalar_sparse_contraction!(
                            Cₜ_2, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, t₀, p
                        )
                    end
                    println()
                end
                println()
            end

            # Temporary store correlators 
            @time "  Write tmp correlators" begin
                PC.write_correlator.(correlator1_file_tmp_arr, correlators1, PC.parms.p_arr)
                PC.write_correlator.(correlator2_file_tmp_arr, correlators2, PC.parms.p_arr)
            end
            println()
        end
        println("\n")
    end

    # Broadcast correlators to all ranks
    @time "Broadcast correlators" begin
        PC.broadcast_correlators!.(correlators1)
        PC.broadcast_correlators!.(correlators2)
    end

    # Store correlators and remove tmp correlators
    ##############################################

    if myrank == 0
        @time "Write correlators" begin
            PC.write_correlator.(correlator1_file_arr, correlators1, PC.parms.p_arr)
            PC.write_correlator.(correlator2_file_arr, correlators2, PC.parms.p_arr)
        end
    end

    @time "Remove tmp correlators" begin
        rm.(correlator1_file_tmp_arr, force=true)
        rm.(correlator2_file_tmp_arr, force=true)
    end

end

main()


# %%