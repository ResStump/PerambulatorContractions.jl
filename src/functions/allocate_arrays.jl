@doc raw"""
    allocate_perambulator() -> τ_αkβlt

Allocate the empty array `τ_αkβlt` with the correct size to store a perambulator.
"""
function allocate_perambulator()
    τ_αkβlt = Array{ComplexF64}(undef, 4, parms.N_modes, 4, parms.N_modes, parms.Nₜ)

    return τ_αkβlt
end

@doc raw"""
    allocate_mode_doublets(mode_doublets_file) -> Φ_kltiₚ

Allocate the empty array `Φ_kltiₚ` with the correct size to store mode doublets.

The HDF5 file path `mode_doublets_file` is used to get the number of momenta stored in the
mode doublets.
"""
function allocate_mode_doublets(mode_doublets_file)
    # Get number of momenta in mode doublets
    Nₚ = length(read_mode_doublets_momenta(mode_doublets_file))

    # Allocate array
    Φ_kltiₚ = Array{ComplexF64}(undef, parms.N_modes, parms.N_modes, parms.Nₜ, Nₚ)

    return Φ_kltiₚ
end

@doc raw"""
    allocate_mode_triplets(mode_triplets_file) -> Φ_Ktiₚ

Allocate the empty array `Φ_Ktiₚ` with the correct size to store mode triplets.

The HDF5 file path `mode_triplets_file` is used to get the number of momenta stored in the
mode triplets.
"""
function allocate_mode_triplets(mode_triplets_file)
    # Get number of momenta in mode triplets
    Nₚ = length(read_mode_triplets_momenta(mode_triplets_file))

    # Allocate array
    K_size = parms.N_modes*(parms.N_modes - 1)*(parms.N_modes - 2) ÷ 6
    Φ_Ktiₚ = Array{ComplexF64}(undef, K_size, parms.Nₜ, Nₚ)

    return Φ_Ktiₚ
end

@doc raw"""
    allocate_sparse_modes(sparse_modes_file=nothing; N_points=nothing)
        -> x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

Allocate the empty arrays `x_sink_μiₓt`, `x_src_μiₓt`, `v_sink_ciₓkt`, `v_src_ciₓkt` with
the correct size to store the sparse spaces and modes.

The number of points in the sparse modes are either determined from the HDF5 file
`sparse_modes_file` or take as `N_points`.
"""
function allocate_sparse_modes(sparse_modes_file=nothing; N_points=nothing)
    if !isnothing(sparse_modes_file)
        # If sparse_modes_file given get number of points in sparse space from it
        file = HDF5.h5open(string(sparse_modes_file), "r")
        _, N_points, _ = size(file["sparse_space_src"])
        close(file)
    elseif isnothing(N_points)
        throw(ArgumentError("either sparse_modes_file or N_points have to be given."))
    end

    # Allocate arrays
    x_sink_μiₓt = Array{Int32}(undef, 3, N_points, parms.Nₜ)
    x_src_μiₓt = Array{Int32}(undef, 3, N_points, parms.Nₜ)
    v_sink_ciₓkt = Array{ComplexF64}(undef, 3, N_points, parms.N_modes, parms.Nₜ)
    v_src_ciₓkt = Array{ComplexF64}(undef, 3, N_points, parms.N_modes, parms.Nₜ)

    return x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt
end
