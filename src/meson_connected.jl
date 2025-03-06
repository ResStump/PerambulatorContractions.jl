# %%########################################################################################
# meson_connected.jl
#
# Compute connected meson correlators from perambulators and mode doublets and sparse modes
# where the contractions are done in parallel using MPI.jl.
#
# Usage:
#   meson_connected.jl -i <parms file> --nranks-per-cnfg <n>
#
# where <parms file> is a toml file containing the required parameters and <n> is the number
# of ranks that simultaneously work on one configuration. If --nranks-per-cnfg is not
# provided, the default value is 1.
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


# %%###############
# Global Parameters
###################

# Set global parameters
PC.read_parameters()

# Split communicators
cnfg_comm, comm_number, my_cnfgs = PC.cnfg_comm()
my_cnfg_rank = MPI.Comm_rank(cnfg_comm)
cnfg_root_comm = MPI.Comm_split(comm, my_cnfg_rank==0, my_cnfg_rank)

# γ-matrices in operator and their labels
Γ_labels = PC.parms_toml["Meson operators"]["Gamma"]
Γ_arr = PC.parse_gamma_string.(Γ_labels)
Nᵧ = length(Γ_arr)

# Correlator matrix type and flavour content
corr_matrix_type = PC.parms_toml["Meson operators"]["corr_matrix_type"]
if !(corr_matrix_type in ["full", "diag"])
    throw(ArgumentError("correlator matrix type not valid! Choose \"full\" or \"diag\""))
end
flavour = PC.parms_toml["Meson operators"]["flavour"]
if !(flavour in ["llbar", "clbar"])
    throw(ArgumentError("flavour not valid! Choose \"llbar\" or \"clbar\""))
end

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(comm_number).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    println("Continuation run\n")
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end

# Use mode doublets or sparse modes?
if PC.parms_toml["Correlator"]["method"] == "sparse"
    method = "sparse"

    # Point separation in spares lattice (will be set later)
    N_sep = nothing
    method_str() = "$(method)_Nsep$(N_sep)"
elseif PC.parms_toml["Correlator"]["method"] == "full"
    method = "full"
    method_str() = method
else
    throw(ArgumentError("correlator method is not valid! Choose \"full\" or \"sparse\""))
end

# Set which momenta should be used
if PC.parms_toml["Momenta"]["p"] == "all"
    p_arr = PC.parms.p_arr
else
    p_arr = PC.parms_toml["Momenta"]["p"]
end

# Momentum indices in mode doublets corresponding to the momentas in p_arr
iₚ_arr = [findfirst(p_ -> p_ == p, PC.parms.p_arr) for p in p_arr]
if (method != "full") && any(isnothing.(iₚ_arr))
    throw(DomainError("a chosen momentum `p` is not contained in the mode doublets."))
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
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"    
corr_file_path() = PC.parms.result_dir/"$(PC.parms_toml["Run name"]["name"])_" *
            "$(PC.parms.N_modes)modes_meson_connected_$(flavour)_$(method_str()).hdf5"
corr_tmp_file_path() = PC.parms.result_dir/"$(PC.parms_toml["Run name"]["name"])_" *
            "$(PC.parms.N_modes)modes_meson_connected_$(flavour)_$(method_str())_" *
            "tmp$(myrank).hdf5"

function write_correlator(; tmp=false)
    if tmp
        file = HDF5.h5open(string(corr_tmp_file_path()), "w")
    else
        file = HDF5.h5open(string(corr_file_path()), "w")
    end

    # Loop over all momentuma
    for (iₚ, p) in enumerate(p_arr)
        p_str = join(p, ",")

        # Write correlator with dimension labels
        if corr_matrix_type == "full"
            file["Correlators/p$p_str"] = C_tnn̄t₀ciₚ[:, :, :, :, :, iₚ]
        elseif corr_matrix_type == "diag"
            file["Correlators/p$p_str"] = C_tnt₀ciₚ[:, :, :, :, iₚ]
        end
        HDF5.attrs(file["Correlators/p$p_str"])["DIMENSION_LABELS"] = labels
    end

    # Write spin structure
    file["Spin Structure/Gamma"] = Γ_labels

    # Write parameter file and program information
    file["parms.toml"] = PC.parms.parms_toml_string
    file["Program Information"] = PC.parms_toml["Program Information"]
    
    close(file)

    return
end


# %%############################
# Generate contraction functions
################################

if method == "sparse"
    Γbar_arr = [PC.γ[4]] .* adjoint.(Γ_arr) .* [PC.γ[4]]
    @time "Generate contraction functions" begin
        if corr_matrix_type == "full"
            contract_arr = Matrix{Function}(undef, Nᵧ, Nᵧ)
            for n in 1:Nᵧ, n̄ in 1:Nᵧ
                contract_arr[n, n̄] = PC.generate_meson_connected_contract_func(
                    Γ_arr[n], Γbar_arr[n̄]
                )
            end
        elseif corr_matrix_type == "diag"
            contract_arr = PC.generate_meson_connected_contract_func.(Γ_arr, Γbar_arr)
        end
    end
    println()
end


# %%#############
# Allocate Arrays
#################

if my_cnfg_rank == 0
    # Select valid cnfg number
    n_cnfg = PC.parms.cnfg_numbers[1]

    # Perambulators
    τ_αkβlt = PC.allocate_perambulator()
    if flavour == "clbar"
        τ_charm_αkβlt = PC.allocate_perambulator()
    end

    # Mode doublets arrays
    if method == "full"
        Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))
    end
    
    # Sparse mode arrays
    if method == "sparse"
        sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

        # Set point separation
        N_points = size(sparse_modes_arrays[1], 2)
        N_sep = (prod(PC.parms.Nₖ)/N_points)^(1/3)
        N_sep = round(Int, N_sep)
    end

    # Correlator and its labels for writing it (order of labels reversed in Julia)
    # (initialize correlators with zeros to use MPI.Reduce for the communication)
    if corr_matrix_type == "full"
        correlator_size = 
            (PC.parms.Nₜ, Nᵧ, Nᵧ, PC.parms.N_src, PC.parms.N_cnfg, length(p_arr))
        C_tnn̄t₀ciₚ = zeros(ComplexF64, correlator_size)
        labels = ["config", "source", "Gamma bar", "Gamma", "t"]
    elseif corr_matrix_type == "diag"
        correlator_size = (PC.parms.Nₜ, Nᵧ, PC.parms.N_src, PC.parms.N_cnfg, length(p_arr))
        C_tnt₀ciₚ = zeros(ComplexF64, correlator_size)
        labels = ["config", "source", "Gamma", "t"]
    end

    # Read correlator file if continuation run
    if continuation_run
        file = HDF5.h5open(string(corr_tmp_file_path()))

        # Loop over all momentuma
        @time "Read tmp file" begin
            for (iₚ, p) in enumerate(p_arr)
                p_str = join(p, ",")
        
                # Write correlator with dimension labels
                if corr_matrix_type == "full"
                    C_tnn̄t₀ciₚ[:, :, :, :, :, iₚ] = read(file["Correlators/p$p_str"])
                elseif corr_matrix_type == "diag"
                    C_tnt₀ciₚ[:, :, :, :, iₚ] = read(file["Correlators/p$p_str"])
                end
            end
        end
        println()

        close(file)
    end
end


# %%#########
# Computation
#############
function compute_contractions!(i_src, t₀, i_cnfg)
    @time "      meson connected contractons" begin
        if my_cnfg_rank == 0
            # Index for source time `t₀`
            i_t₀ = t₀+1

            # Time range to be computed
            iₜ_range = i_t₀ .+ (0:PC.parms.Nₜ-1)
            iₜ_range = mod1.(iₜ_range, PC.parms.Nₜ)

            # Convert arrays to vectors of arrays in the time axis
            if flavour == "clbar"
                τ₁_arr = eachslice(τ_charm_αkβlt, dims=5)[iₜ_range]
                τ₂_arr = eachslice(τ_αkβlt, dims=5)[iₜ_range]
            elseif flavour == "llbar"
                τ₁_arr = eachslice(τ_αkβlt, dims=5)[iₜ_range]
                τ₂_arr = τ₁_arr
            end
            if method == "full"
                Φ_arr = eachslice(Φ_kltiₚ, dims=3)[iₜ_range]

                # Select source time `t₀`
                Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]

                all_arrays = (τ₁_arr, τ₂_arr, Φ_arr, [Φ_kliₚ_t₀])
            elseif method == "sparse"
                # Unpack sparse modes arrays
                x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

                x_sink_arr = eachslice(x_sink_μiₓt, dims=3)[iₜ_range]
                v_sink_arr = eachslice(v_sink_ciₓkt, dims=4)[iₜ_range]

                # Select source time `t₀`
                x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
                v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

                all_arrays = (τ₁_arr, τ₂_arr, x_sink_arr, [x_src_μiₓ_t₀],
                              v_sink_arr, [v_src_ciₓk_t₀])
            end
        end

        # Function to compute contractions
        if method == "full"
            contractions = (τ₁, τ₂, Φ_t, Φ_t₀) -> begin
                C_arr = []

                # Loop over all momentum indices
                for iₚ in iₚ_arr
                    C = PC.meson_connected_contractions(
                        τ₁, τ₂, Φ_t, Φ_t₀, Γ_arr, iₚ, corr_matrix_type=="full"
                    )
                    push!(C_arr, C)
                end

                # Return as contiguous arrays
                return stack(C_arr)
            end
        elseif method == "sparse"
            contractions = (τ₁, τ₂, x_sink, x_src, v_sink, v_src) -> begin
                C = PC.meson_connected_sparse_contractions(
                    τ₁, τ₂, (x_sink, x_src, v_sink, v_src), p_arr, contract_arr,
                    corr_matrix_type=="full"
                )

                return C
            end
        end

        # Distribute workload and compute contraction
        if my_cnfg_rank == 0
            corr_arr = PC.mpi_broadcast(contractions, all_arrays..., comm=cnfg_comm,
                                        log_prefix="      ")
        else
            PC.mpi_broadcast(contractions, comm=cnfg_comm)
        end

        # Store correlator entries
        if my_cnfg_rank == 0
            for (iₜ, corr_t) in enumerate(corr_arr)
                if corr_matrix_type == "full"
                    C_tnn̄t₀ciₚ[iₜ, :, :, i_src, i_cnfg, :] = corr_t
                elseif corr_matrix_type == "diag"
                    C_tnt₀ciₚ[iₜ, :, i_src, i_cnfg, :] = corr_t
                end
            end
        end
    end
end


function main()
    # Loop over all configurations
    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_numbers)
        # Skip the cnfgs this rank doesn't have to compute
        if n_cnfg ∉ my_cnfgs
            continue
        end
        if continuation_run && (n_cnfg in finished_cnfgs)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            if my_cnfg_rank == 0 && method == "full"
                @time "  Read mode doublets" begin
                    PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
                end
            elseif my_cnfg_rank == 0 && method == "sparse"
                @time "  Read sparse modes " begin
                    PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
                end
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                if my_cnfg_rank == 0
                    @time "    Read perambulators" begin
                        PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                        if flavour == "clbar"
                            PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                                  τ_charm_αkβlt)
                        end
                    end
                end
                println()

                compute_contractions!(i_src, t₀, i_cnfg)

                println()
            end

            # Temporary store correlators and update finished_cnfgs
            if my_cnfg_rank == 0
                @time "  Write tmp Files" begin
                    write_correlator(tmp=true)

                    push!(finished_cnfgs, n_cnfg)
                    DF.writedlm(string(finished_cnfgs_file), finished_cnfgs, '\n')
                end
            end
                
            # Run garbage collector for "young" objects
            GC.gc(false)
            println()
        end

        # Run garbage collector (fully)
        GC.gc()

        println("\n")
    end

    # Wait until all ranks finished
    MPI.Barrier(comm)

    if my_cnfg_rank == 0
        @time "Send correlators to root" begin
            if corr_matrix_type == "full"
                MPI.Reduce!(C_tnn̄t₀ciₚ, MPI.SUM, cnfg_root_comm, root=0)
            elseif corr_matrix_type == "diag"
                MPI.Reduce!(C_tnt₀ciₚ, MPI.SUM, cnfg_root_comm, root=0)
            end
        end
    end

    if myrank == 0
        @time "Write correlator" begin
            write_correlator()
        end
    end

    # Remove finished_cnfgs and tmp correlator file
    if my_cnfg_rank == 0
        rm(finished_cnfgs_file, force=true)
        rm(corr_tmp_file_path(), force=true)
    end

    println("Program finished successfully.")
end

main()

# %%
