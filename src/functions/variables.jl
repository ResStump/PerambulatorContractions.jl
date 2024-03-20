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
σ = Dict(0=>σ₀, 1=>σ₁, 2=>σ₂, 3=>σ₃)

# Euclidean gamma matrice in chiral rep
γ₁ = [   O -im*σ₁;
      im*σ₁    O]
γ₂ = [   O -im*σ₂;
      im*σ₂    O]
γ₃ = [   O -im*σ₃;
      im*σ₃    O]
γ₄ = [ O  -σ₀;
      -σ₀  O]
γ₅ = [σ₀  O;
      O  -σ₀]

γ = [γ₁, γ₂, γ₃, γ₄, γ₅]

C = im*γ₂*γ₄