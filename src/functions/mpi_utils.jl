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