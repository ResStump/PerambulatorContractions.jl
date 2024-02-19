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
import TensorOperations as TO
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import BenchmarkTools.@btime
import Startup

include("IO.jl")
include("contractions.jl")

# Add infile manually to arguments
# pushfirst!(ARGS, "-i", "run_pseudoscalar/input/pseudoscalar_B450r000.toml")


# %%###############
# Global Parameters
###################

# Instance of Parms
parms = nothing

# Dict with parameters from toml
parms_toml = Dict()



# %%#########
# Computation
#############

read_parameters()

# Compute Contractions
######################

# File paths
perambulator_file(n_cnfg, i_src) = parms.perambulator_dir/"perambulator_" *
                                   "$(parms_toml["Perambulator"]["label_base"])$(i_src)_" *
                                   "$(parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = parms.mode_doublets_dir/
                             "mode_doublets_$(parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = parms.sparse_modes_dir/
                            "sparse_modes_$(parms_toml["Run name"]["name"])n$(n_cnfg)"





# Compare imported mode doublets to those computed from Laplacian eigenmodes
############################################################################

# Read mode doublets
#= Φ_klt_iₚ = read_mode_doublets(mode_doublets_file(1), 2)[:, :, :, iₚ]


# Read eigen modese and the positions
sparse_modes_arrays = read_sparse_modes(sparse_modes_file(1), 2)
x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

x_sink_μiₓ
x_src_μiₓt
v_sink_ciₓkt
v_src_ciₓkt


# Number of lattice points
_, N_points = size(x_sink_μiₓ)


# Create empty mode doublet Array
Φ_klt_iₚ_ = Array{ComplexF64, 3}(undef, parms.N_modes, parms.N_modes, parms.Nₜ)
Φ_klt_iₚ_adj = Array{ComplexF64, 3}(undef, parms.N_modes, parms.N_modes, parms.Nₜ)

for t in 1:parms.Nₜ
    # Laplace modes and mode doublets at time t
    v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, t]
    Φ_kl_tiₚ = @view Φ_klt_iₚ[:, :, t]
    Φ_kl_tiₚ_ = @view Φ_klt_iₚ_[:, :, t]
    Φ_kl_tiₚ_adj = @view Φ_klt_iₚ_adj[:, :, t]

    # Compute exp(±ipx) and reshape it to match shape of v_sink_ciₓk_t
    exp_mipx_sink_iₓ = exp.(-2π*im*(x_sink_μiₓ./parms.Nₖ)'*p)
    exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))

    TO.@tensoropt begin
        Φ_kl_tiₚ_[k, l] = conj(v_sink_ciₓk_t[c, iₓ, k]) * 
                          (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[c, iₓ, l]
    end

    TO.@tensoropt begin
        Φ_kl_tiₚ_adj[k, l] = (v_sink_ciₓk_t[c, iₓ, l]) * 
                          conj((exp_mipx_sink_iₓ) .* v_sink_ciₓk_t)[c, iₓ, k]
    end

    @assert Φ_kl_tiₚ' ≈ Φ_kl_tiₚ_adj
end

Φ_klt_iₚ ≈ Φ_klt_iₚ_
Φ_klt_iₚ ≈ Φ_klt_iₚ_adj =#



# Now follows the actual computation
# %%




# Get momentum index
p_arr = read_mode_doublet_momenta(mode_doublets_file(parms.cnfg_indices[1]))
iₚ = findfirst(p -> p == parms.p, eachrow(p_arr))
if isnothing(iₚ)
    throw(DomainError("the chosen momentum 'p' is not contained in mode doublets."))
end

# Allocate arrays to store the correlator
correlator = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator2 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator3 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)

for (i_cnfg, n_cnfg) in enumerate(parms.cnfg_indices)
    println("Configuration $n_cnfg")
    @time "Finished configuration $n_cnfg" begin
        @time "  Read sparse modes " begin
            sparse_modes_arrays = read_sparse_modes(sparse_modes_file(n_cnfg))
        end
        @time "  Read mode doublets" begin
            Φ_kltiₚ = read_mode_doublets(mode_doublets_file(n_cnfg))
        end
        println()

        for (i_src, t₀) in enumerate(parms.tsrc_arr[i_cnfg, :])
            println("  Source: $i_src of $(parms.N_src)")

            @time "    Read perambulator" begin
                τ_αkβlt = read_perambulator(perambulator_file(n_cnfg, t₀))
            end
            println()

            Cₜ = @view correlator[:, i_src, i_cnfg]
            Cₜ_2 = @view correlator2[:, i_src, i_cnfg]
            Cₜ_3 = @view correlator3[:, i_src, i_cnfg]
            @time "    pseudoscalar_contraction!       " begin
                pseudoscalar_contraction!(Cₜ, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
            end
            @time "    pseudoscalar_contraction_p0!    " begin
                pseudoscalar_contraction_p0!(Cₜ_2, τ_αkβlt, t₀)
            end
            @time "    pseudoscalar_sparse_contraction!" begin
                pseudoscalar_sparse_contraction!(Cₜ_3, τ_αkβlt, sparse_modes_arrays, t₀,
                                                 parms.p)
            end
            println()
        end
    end
    println()
end


# Store correlator
##################

run_name = parms_toml["Run name"]["name"]

correlator_file = "$(run_name)_" * "$(parms.N_modes)modes_pseudoscalar.hdf5"
correlator2_file = "$(run_name)_" * "$(parms.N_modes)modes_pseudoscalar_p0.hdf5"
correlator3_file = "$(run_name)_" * "$(parms.N_modes)modes_pseudoscalar_sparse.hdf5"

@time "Write correlators" begin
    write_correlator(parms.result_dir/correlator_file, correlator)
    write_correlator(parms.result_dir/correlator2_file, correlator2, zeros(Int, 3))
    write_correlator(parms.result_dir/correlator3_file, correlator3)
end


# %%

#= #= correlator = HDF5.h5read("run_pseudoscalar_juwels_tmp0/program_files/results/B450r000_32modes_pseudoscalar.hdf5", "Correlator")
correlator2 = HDF5.h5read("run_pseudoscalar_juwels_tmp0/program_files/results/B450r000_32modes_pseudoscalar_p0.hdf5", "Correlator")
correlator3 = HDF5.h5read("run_pseudoscalar_juwels_tmp0/program_files/results/B450r000_32modes_pseudoscalar_sparse.hdf5", "Correlator") =#

Nₜ, _, _ = size(correlator)



# %%

import Plots as Plt
import Statistics as Stats
using LaTeXStrings

corr = vec(Stats.mean(real(correlator), dims=(2, 3)))
corr[corr.<=0] .= NaN
corr2 = vec(Stats.mean(real(correlator2), dims=(2, 3)))
corr2[corr2.<=0] .= NaN
corr3 = vec(Stats.mean(real(correlator3), dims=(2, 3)))
corr3[corr3.<=0] .= NaN

corr_ = Stats.mean(real(correlator), dims=(2))
corr_[corr_.<=0] .= NaN
corr2_ = Stats.mean(real(correlator2), dims=(2))
corr2_[corr2_.<=0] .= NaN

plot = Plt.plot(xlabel=L"t/a", ylabel=L"C(t)", yscale=:log)
Plt.scatter!(1:Nₜ, corr, label="Using mode doublets")
#Plt.scatter!(1:Nₜ, corr2, label="For zero momentum")
Plt.scatter!(1:Nₜ, corr3, label="Position space sampling")
for i in 1:parms.N_cnfg
    #= Plt.plot!(legend=false)
    Plt.scatter!(1:Nₜ, corr_[:, 1, i][corr_[:, 1, i].>0.0], label="Using mode doublets")
    #Plt.scatter!(1:Nₜ, corr2_[:, 1, i][corr2_[:, 1, i].>0.0], label="Using full eigenvectors") =#

end
display(plot)

# Plt.savefig(p, "pseudoscalar_p1,0,0_Nsep1.pdf") =#


# %%