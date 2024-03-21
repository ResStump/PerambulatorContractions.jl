# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()


# Paths to files
perambulator_file = PC.parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
sparse_modes_file = PC.parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"

# Read files
τ_αkβlt = PC.read_perambulator(perambulator_file)
p_arr = PC.read_mode_doublet_momenta(mode_doublets_file)
Φ_kltiₚ = PC.read_mode_doublets(mode_doublets_file)
sparse_modes_arrays = PC.read_sparse_modes(sparse_modes_file)


@testset "Compare full modes at sink and src" begin
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Loop over all positions at sink
    for (iₓ_sink, x_sink) in enumerate(eachcol(x_sink_μiₓ))
        for iₜ in 1:PC.parms.Nₜ
            iₓ_src_t = findfirst(x -> x == x_sink, eachcol(x_src_μiₓt[:, :, iₜ]))

            @test v_sink_ciₓkt[:, iₓ_sink, :, iₜ] == v_src_ciₓkt[:, iₓ_src_t, :, iₜ]
        end
    end

end

@testset "Compare mode doublets from full modes to those from file" begin
    Φ2_kltiₚ = similar(Φ_kltiₚ)
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Loop over all momenta
    for (iₚ, p) in enumerate(eachrow(p_arr))
        # Loop over all sink time indice
        for iₜ in 1:PC.parms.Nₜ
            # Laplace modes at sink time t (index 'iₜ')
            v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

            # Compute exp(+ipx) and reshape it to match shape of Laplace modes
            exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./PC.parms.Nₖ)'*p)
            exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))

            Φ2_kl_tiₚ = @view Φ2_kltiₚ[:, :, iₜ, iₚ]
            TO.@tensoropt begin
                Φ2_kl_tiₚ[k, l] = conj(v_sink_ciₓk_t[a, iₓ, k]) * 
                    (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ, l]
            end
        end
    end

    # Check if both mode doublets are the same
    @test Φ_kltiₚ ≈ Φ2_kltiₚ
end

@testset "Compare pseudoscalar correlator from mode doublets and from full modes" begin
    Cₜ_mode_doublets = Array{ComplexF64}(undef, PC.parms.Nₜ)
    Cₜ_mode_doublets_p0 = similar(Cₜ_mode_doublets)
    Cₜ_full_modes = similar(Cₜ_mode_doublets)

    # Source time t₀
    t₀ = PC.parms.tsrc_arr[1, 1]

    # For zero mometum
    PC.pseudoscalar_contraction_p0!(Cₜ_mode_doublets_p0, τ_αkβlt, t₀)

    # Loop over all momenta
    for (iₚ, p) in enumerate(eachrow(p_arr))
        PC.pseudoscalar_contraction!(Cₜ_mode_doublets, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
        PC.pseudoscalar_sparse_contraction!(Cₜ_full_modes, τ_αkβlt, sparse_modes_arrays,
                                            t₀, p)

        @test Cₜ_mode_doublets ≈ Cₜ_full_modes

        if p == [0, 0, 0]
            @test Cₜ_mode_doublets ≈ Cₜ_mode_doublets_p0
        end
    end
end

@testset "Compare pseudoscalar contractions with one and two perambulators" begin
    Cₜ_1 = Array{ComplexF64}(undef, PC.parms.Nₜ)
    Cₜ_2 = similar(Cₜ_1)

    # Source time t₀ and momentum index iₚ
    t₀ = PC.parms.tsrc_arr[1, 1]
    iₚ = 1
    p = p_arr[iₚ, :]

    # For zero momentum
    PC.pseudoscalar_contraction_p0!(Cₜ_1, τ_αkβlt, t₀)
    PC.pseudoscalar_contraction_p0!(Cₜ_2, τ_αkβlt, τ_αkβlt, t₀)
    @test Cₜ_1 ≈ Cₜ_2

    # Using mode doublets
    PC.pseudoscalar_contraction!(Cₜ_1, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
    PC.pseudoscalar_contraction!(Cₜ_2, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
    @test Cₜ_1 ≈ Cₜ_2

    # Using sparse modes
    PC.pseudoscalar_sparse_contraction!(Cₜ_1, τ_αkβlt, sparse_modes_arrays, t₀, p)
    PC.pseudoscalar_sparse_contraction!(Cₜ_2, τ_αkβlt, τ_αkβlt, sparse_modes_arrays, t₀, p)
    @test Cₜ_1 ≈ Cₜ_2
end

@testset "Meson_connected contractions" begin
    Cₜ_1 = Array{ComplexF64}(undef, PC.parms.Nₜ)
    Cₜ_2 = similar(Cₜ_1)

    # Source time t₀ and momentum index iₚ
    t₀ = PC.parms.tsrc_arr[1, 1]
    iₚ = 1
    p = p_arr[iₚ, :]

    # Matrices in interpolstors
    Γ_arr = [PC.γ[1], PC.γ[2], PC.γ[3], PC.γ[5]*PC.γ[4], PC.γ[5]]

    for Γ in Γ_arr
        Γbar = PC.γ[4]*Γ*PC.γ[4]

        # Zero Momentum
        ###############

        # Direct computation
        for iₜ in 1:PC.parms.Nₜ
            # perambulators at sink (index `iₜ`)
            τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]

            # Tensor contraction
            TO.@tensoropt begin
                C = (PC.γ[5]*Γ)[α, α'] * τ_αkβl_t[α', k, β, l] *
                    (-Γbar*PC.γ[5])[β, β'] * conj(τ_αkβl_t[α, k, β', l])
            end

            # Circularly shift time such that t₀=0
            Cₜ_1[mod1(iₜ-t₀, PC.parms.Nₜ)] = C
        end
        
        PC.meson_connected_contraction_p0!(Cₜ_2, τ_αkβlt, τ_αkβlt, Γ, Γbar, t₀)
        @test Cₜ_1 ≈ Cₜ_2


        # Non-Zero Momentum
        ###################

        # Index for source time `t₀`
        i_t₀ = t₀+1

        # Mode doublet at source time `t₀`
        Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]

        # Direct computation
        for iₜ in 1:PC.parms.Nₜ
            # Mode doublet and perambulators at sink time t (index `iₜ`)
            Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
            τ_αkβl_t = @view τ_αkβlt[:, :, :, :,iₜ]
            
            # Tensor contraction
            TO.@tensoropt begin
                C = Φ_kl_tiₚ[k, k'] *
                    (PC.γ[5]*Γ)[α, α'] * τ_αkβl_t[α', k', β, l'] *
                    conj(Φ_kl_t₀iₚ[l, l']) *
                    (-Γbar*PC.γ[5])[β, β'] * conj(τ_αkβl_t[α, k, β', l])
            end

            # Circularly shift time such that t₀=0
            Cₜ_1[mod1(iₜ-t₀, PC.parms.Nₜ)] = C
        end
        
        PC.meson_connected_contraction!(Cₜ_2, τ_αkβlt, τ_αkβlt, Φ_kltiₚ,
                                        Γ, Γbar, t₀, iₚ)
        @test Cₜ_1 ≈ Cₜ_2

        # Check if meson_connected_sparse_contraction! is also correct
        PC.meson_connected_sparse_contraction!(
            Cₜ_2, τ_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ, Γbar, t₀, p
        )
        @test Cₜ_1 ≈ Cₜ_2
    end
end

@testset "Compare meson_connected and pseudoscalar contractions" begin
    Cₜ_pseudoscalar = Array{ComplexF64}(undef, PC.parms.Nₜ)
    Cₜ_meson_connected = similar(Cₜ_pseudoscalar)

    # Source time t₀ and momentum index iₚ
    t₀ = PC.parms.tsrc_arr[1, 1]
    iₚ = 1
    p = p_arr[iₚ, :]

    # Matrices in interpolstors
    Γ, Γbar = PC.γ[5], -PC.γ[5]

    # For zero momentum
    PC.pseudoscalar_contraction_p0!(Cₜ_pseudoscalar, τ_αkβlt, τ_αkβlt, t₀)
    PC.meson_connected_contraction_p0!(Cₜ_meson_connected, τ_αkβlt, τ_αkβlt,
                                       Γ, Γbar, t₀)
    @test Cₜ_pseudoscalar ≈ Cₜ_meson_connected

    # Using mode doublets
    PC.pseudoscalar_contraction!(Cₜ_pseudoscalar, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
    PC.meson_connected_contraction!(Cₜ_meson_connected, τ_αkβlt, τ_αkβlt, Φ_kltiₚ,
                                    Γ, Γbar, t₀, iₚ)
    @test Cₜ_pseudoscalar ≈ Cₜ_meson_connected

    # Using sparse modes
    PC.pseudoscalar_sparse_contraction!(Cₜ_pseudoscalar, τ_αkβlt, τ_αkβlt,
                                        sparse_modes_arrays, t₀, p)
    PC.meson_connected_sparse_contraction!(Cₜ_meson_connected, τ_αkβlt, τ_αkβlt,
                                           sparse_modes_arrays, Γ, Γbar, t₀, p)
    @test Cₜ_pseudoscalar ≈ Cₜ_meson_connected
end
