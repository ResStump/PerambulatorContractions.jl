@doc raw"""
    cnfg_comm() -> cnfg_comm::MPI.comm, comm_number::Int, my_cnfgs::Vector{Int}

Split the MPI global communicator `MPI.COMM_WORLD` into subcommunicators `cnfg_comm`
consisting of `parms.N_ranks_per_cnfg` ranks which simultaneosly work on a configuration.
These communicators contain consecutive ranks. Each communicator gets a number `comm_number`
which is also returned. \
Additionally, return the configurations this rank has to work on.
"""
function cnfg_comm()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    N_ranks = MPI.Comm_size(comm)

    if mod(N_ranks, parms.N_ranks_per_cnfg) != 0
        throw(ArgumentError("the number of ranks must be divisible by the number of "*
                            "ranks per cnfg."))
    end

    # New communicator (consecutive ranks grouped together)
    cnfg_comm = MPI.Comm_split(comm, myrank÷parms.N_ranks_per_cnfg, myrank)

    # Assign numbers to the communicators to get number of cnfgs per communicator
    comm_number = myrank÷parms.N_ranks_per_cnfg
    N_comm = N_ranks÷parms.N_ranks_per_cnfg
    N_cnfg_per_comm_max = ceil(Int, parms.N_cnfg/N_comm)

    # Configurations for this rank
    my_cnfg_indices = (0:parms.N_cnfg-1) .÷ N_cnfg_per_comm_max .== comm_number
    my_cnfgs = parms.cnfg_numbers[my_cnfg_indices]
    
    return cnfg_comm, comm_number, my_cnfgs
end

@doc raw"""
    broadcast_correlators!(correlator, cnfg_dim=3)

Broadcast `correlator` to all ranks by assuming that each rank computed that part of
`correlator` for which `is_my_cnfg(i_cnfg) == true`. The integer `cnfg_dim` specifies which
dimension in `correlator` is the configuration index (default is 3).
"""
function broadcast_correlators!(correlator, cnfg_dim=3)
    # Check size of correlator
    N_cnfg = size(correlator)[cnfg_dim]
    if N_cnfg != parms.N_cnfg
        throw(ArgumentError("the correlator doesn't have the right number of "*
                            "configurations."))
    end

    comm = MPI.COMM_WORLD
    N_ranks = MPI.Comm_size(comm)

    N_cnfg_per_rank = ceil(Int, parms.N_cnfg/N_ranks)
    for rank in 0:N_ranks-1
        # Determine start and end of local correlator data
        first_cnfg = 1 + N_cnfg_per_rank*rank
        last_cnfg = min(N_cnfg_per_rank*(rank+1), parms.N_cnfg)

        MPI.Bcast!(selectdim(correlator, cnfg_dim, first_cnfg:last_cnfg), rank, comm)
    end
end

@doc raw"""
    send_correlator_to_root!(correlator, cnfg_dim=3)

Send `correlator` to root (rank 0) by assuming that each rank computed that part of
`correlator` for which `is_my_cnfg(i_cnfg) == true`. The integer `cnfg_dim` specifies which
    dimension in `correlator` is the configuration index (default is 3).
"""
function send_correlator_to_root!(correlator, cnfg_dim=3)
    # Check size of correlator
    N_cnfg = size(correlator)[cnfg_dim]
    if N_cnfg != parms.N_cnfg
        throw(ArgumentError("the correlator doesn't have the right number of "*
                            "configurations."))
    end

    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    N_ranks = MPI.Comm_size(comm)

    N_cnfg_per_rank = ceil(Int, parms.N_cnfg/N_ranks)
    for rank in 1:N_ranks-1
        # Determine start and end of local correlator
        first_cnfg = 1 + N_cnfg_per_rank*rank
        last_cnfg = min(N_cnfg_per_rank*(rank+1), parms.N_cnfg)
        
        correlator_view = selectdim(correlator, cnfg_dim, first_cnfg:last_cnfg)
        if myrank == rank
            MPI.Send(correlator_view, comm, dest=0)
        elseif myrank == 0
            MPI.Recv!(correlator_view, comm, source=rank)
        end
    end
end

@doc raw"""
    mpi_broadcast(f, vectors::AbstractVector{<:AbstractArray}...; comm=MPI.COMM_WORLD, root::Int=0, log_prefix="mpi_broadcast: ")

Broadcast the function `f` over the vectors of arrays `vectors` using MPI on the 
communicator `comm`. The arrays in `vectors` are distributed among the available ranks which
process a subset of them. Then, the result is gathered on rank `root` and returned. 
This function has the same behavior as f.(vectors...) regarding the lengths of the
`vectors`.

# Usage
On rank `root`, call `mpi_broadcast(f, vectors...)` and on the remaining ranks, call
`mpi_broadcast(f)` (using additional arguments if necessary).

# Performance tips
To acheive good performance, `length(vectors)` should be divisible by
`MPI.Comm_size(comm)`. \
The output of `f` can be of arbitrary type, therefore the gather communication is done
using serialization which might be slow for large arrays.
"""
function mpi_broadcast(f, vectors::AbstractVector{<:AbstractArray}...; comm=MPI.COMM_WORLD,
                       root::Int=0, log_prefix="mpi_broadcast: ")
    myrank = MPI.Comm_rank(comm)
    N_ranks = MPI.Comm_size(comm)
    nonroot_ranks = deleteat!(collect(0:N_ranks-1), root+1)

    # Check input sizes on root
    if myrank == root
        if length(vectors) < 1
            throw(ArgumentError("at least one vector must be provided on root"))
        end

        # Check that all vectors have correct lengths
        v_lengths = length.(vectors)
        N_elem = maximum(v_lengths)
        for v in vectors
            if !(length(v) in [1, N_elem])
                throw(ArgumentError("vectors could not be broadcast to a common length"))
            end
        end

        # Get size and element types of the arrays and check their consistency
        types = []
        sizes = []
        for v in vectors
            push!(types, eltype(v[1]))
            push!(sizes, size(v[1]))
        end

        for (i, v) in enumerate(vectors)
            for a in v[2:end]
                eltype(a) == types[i] ||
                    throw(ArgumentError("all arrays in one vector must have the same "*
                                        "element type"))
                size(a) == sizes[i] ||
                    throw(ArgumentError("all arrays in one vector must have the same size"))
            end
        end
    else
        if length(vectors) != 0
            throw(ArgumentError("only root rank takes vectors as input"))
        end

        v_lengths = nothing
        N_elem = nothing
        types = nothing
        sizes = nothing
    end

    @time "$(log_prefix)Distribute data" begin
        v_lengths, types, sizes = MPI.bcast((v_lengths, types, sizes), comm, root=root)
        N_elem = maximum(v_lengths)

        # Distribution of work among ranks (each rank processes consecutive elements)
        N_elem_per_rank_max = ceil(Int, N_elem/N_ranks)
        first = myrank*N_elem_per_rank_max + 1
        last = min((myrank+1)*N_elem_per_rank_max, N_elem)
        N_elem_loc = last - first + 1
        v_lengths_loc = min.(v_lengths, N_elem_loc) # Length of the local vectors

        # Allocate memory for local vectors
        vectors_loc = [Vector{Array{type}}(undef, v_len)
                    for (type, v_len) in zip(types, v_lengths_loc)]
        if myrank != root
            for idx in eachindex(vectors_loc)
                for i in eachindex(vectors_loc[idx])
                    vectors_loc[idx][i] = Array{types[idx]}(undef, sizes[idx])
                end
            end
        end
            
        # Distribute data
        rreq_arr = MPI.Request[]
        sreq_arr = MPI.Request[]
        MPI.Barrier(comm)
        if myrank != root
            # Receive on rank != root
            for idx in eachindex(vectors_loc)
                for i in eachindex(vectors_loc[idx])
                    rreq = MPI.Irecv!(vectors_loc[idx][i], comm; source=root, tag=myrank)
                    push!(rreq_arr, rreq)
                end
            end
        else
            # Send from root
            for rank in nonroot_ranks
                first_r = rank*N_elem_per_rank_max + 1
                last_r = min((rank+1)*N_elem_per_rank_max, N_elem)
                for v in vectors
                    if length(v) == 1
                        sreq = MPI.Isend(v[1], comm; dest=rank, tag=rank)
                        push!(sreq_arr, sreq)
                    else
                        for a in v[first_r:last_r]
                            sreq = MPI.Isend(a, comm; dest=rank, tag=rank)
                            push!(sreq_arr, sreq)
                        end
                    end
                end
            end

            # Data on root
            for idx in eachindex(vectors)
                if length(vectors[idx]) == 1
                    vectors_loc[idx][1] = vectors[idx][1]
                else
                    vectors_loc[idx][:] = vectors[idx][first:last]
                end
            end
        end
        MPI.Waitall(vcat(rreq_arr, sreq_arr))
        MPI.Barrier(comm)
    end

    # Broadcast
    @time "$(log_prefix)Computation" result_loc = f.(vectors_loc...)

    # Allocate result array on root
    if myrank == root
        # Allocate result memory
        result = Vector{eltype(result_loc)}(undef, N_elem)

        # Result from root
        result[first:last] = result_loc
    end

    # Gather results on root
    @time "$(log_prefix)Waiting for other ranks to finish" MPI.Barrier(comm)
    @time "$(log_prefix)Gather results" begin
        if myrank != root
            MPI.send(result_loc, comm; dest=root, tag=myrank)
        else
            for rank in nonroot_ranks
                first_r = rank*N_elem_per_rank_max + 1
                last_r = min((rank+1)*N_elem_per_rank_max, N_elem)
                result[first_r:last_r] = MPI.recv(comm; source=rank, tag=rank)
            end
        end
        MPI.Barrier(comm)
    end

    if myrank == root
        return result
    else
        return
    end
end
