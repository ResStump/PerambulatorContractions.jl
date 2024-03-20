anti_commutator(A, B) = A*B + B*A

@testset "Gamma matrices" begin
    # Metric
    δ = LA.Diagonal([1, 1, 1, 1])

    # Anti-communtion relations
    for μ in 1:4, ν in μ:4
        @test anti_commutator(PC.γ[μ], PC.γ[ν]) == LA.I*2δ[μ, ν]
    end

    # anti-commutation relations with γ₅
    for μ in 1:4
        @test anti_commutator(PC.γ[5], PC.γ[μ]) == zeros(4, 4)
    end

    # Inverse and ajoint
    for μ in 1:5
        @test PC.γ[μ] == PC.γ[μ]^-1 == PC.γ[μ]'
    end

    # Charge conjugate matrix
    for μ in 1:4
        @test PC.C*PC.γ[μ]*PC.C^-1 == -transpose(PC.γ[μ])
    end
    @test PC.C == PC.C^-1 == PC.C' == -transpose(PC.C)
end