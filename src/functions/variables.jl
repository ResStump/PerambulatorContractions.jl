# Identity matrix
I = LA.diagm(ComplexF64[1.0, 1.0, 1.0, 1.0])

# Pauli matrices
σ₀ = [1 0;
      0 1]
σ₁ = [0 1;
      1 0]
σ₂ = [ 0 -im;
      im  0]
σ₃ = [1  0;
      0 -1]
O = [0 0;
     0 0]
σ = [σ₁, σ₂, σ₃]

# Euclidean gamma matrice in chiral rep
γ₁ = ComplexF64[   O -im*σ₁;
                im*σ₁    O]
γ₂ = ComplexF64[   O -im*σ₂;
                im*σ₂    O]
γ₃ = ComplexF64[   O -im*σ₃;
                im*σ₃    O]
γ₄ = ComplexF64[ O  -σ₀;
                -σ₀  O]
γ₅ = ComplexF64[σ₀  O;
                O  -σ₀]

γ = [γ₁, γ₂, γ₃, γ₄, γ₅]

# Charge conjugation matrix
C = im*γ₂*γ₄

# Commutator of gamma matrices
σ_μν(μ, ν) = 0.5im*(γ[μ]*γ[ν] - γ[ν]*γ[μ])

# Positive/negative parity projectors
Pp = 0.5*(I + γ₄)
Pm = 0.5*(I - γ₄)

# Basis transformation to non-relativistic basis
U = 1/sqrt(2)*(LA.I + γ[5]*γ[4])
