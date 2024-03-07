import MPI

@doc raw"""
    is_my_cnfg(i_cnfg)

Determine if the configuration with index `i_cnfg` has to be computed on this rank.

The configuration are distributed among the ranks such that each rank processes consecutive
configuration.
"""
function is_my_cnfg(i_cnfg)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    N_ranks = MPI.Comm_size(MPI.COMM_WORLD)

    N_cnfg_per_rank = ceil(Int, parms.N_cnfg/N_ranks)
    
    return (i_cnfg-1)÷N_cnfg_per_rank == myrank
end

@doc raw"""
    broadcast_correlators!(correlator)

Broadcast `correlator` to all ranks by assuming that each rank computed that part of
`correlator` for which `is_my_cnfg(i_cnfg) == true`.
"""
function broadcast_correlators!(correlator)
    # Check size of correlator
    Nₜ, N_src, N_cnfg = size(correlator)
    if Nₜ != parms.Nₜ || N_src != parms.N_src || N_cnfg != parms.N_cnfg
        throw(ArgumentError("the correlator doesn't have the correct size."))
    end

    comm = MPI.COMM_WORLD
    N_ranks = MPI.Comm_size(comm)

    N_cnfg_per_rank = ceil(Int, parms.N_cnfg/N_ranks)
    for rank in 0:N_ranks-1
        # Determine start and end of local correlator
        first_cnfg = 1 + N_cnfg_per_rank*rank
        last_cnfg = min(N_cnfg_per_rank*(rank+1), parms.N_cnfg)

        MPI.Bcast!(@view(correlator[:, :, first_cnfg:last_cnfg]), rank, comm)
    end
end

@doc raw"""
    send_correlator_to_root!(correlator)

Send `correlator` to root (rank 0) by assuming that each rank computed that part of
`correlator` for which `is_my_cnfg(i_cnfg) == true`.
"""
function send_correlator_to_root!(correlator)
    # Check size of correlator
    Nₜ, N_src, N_cnfg = size(correlator)
    if Nₜ != parms.Nₜ || N_src != parms.N_src || N_cnfg != parms.N_cnfg
        throw(ArgumentError("the correlator doesn't have the correct size."))
    end

    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    N_ranks = MPI.Comm_size(comm)

    N_cnfg_per_rank = ceil(Int, parms.N_cnfg/N_ranks)
    for rank in 1:N_ranks-1
        # Determine start and end of local correlator
        first_cnfg = 1 + N_cnfg_per_rank*rank
        last_cnfg = min(N_cnfg_per_rank*(rank+1), parms.N_cnfg)
        
        correlator_view = @view(correlator[:, :, first_cnfg:last_cnfg])
        if myrank == rank
            MPI.Send(correlator_view, comm, dest=0)
        elseif myrank == 0
            MPI.Recv!(correlator_view, comm, source=rank)
        end
    end
end