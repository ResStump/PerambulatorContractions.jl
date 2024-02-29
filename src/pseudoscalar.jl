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
import Random
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import BenchmarkTools.@btime
import Startup

include("allocate_arrays.jl")
include("contractions.jl")
include("IO.jl")
include("utils.jl")

# Add infile manually to arguments
# pushfirst!(ARGS, "-i", "run_pseudoscalar/input/pseudoscalar_16x8v1.toml")


# %%#############################
# Global Parameters and Functions
#################################

# Instance of Parms
parms = nothing

# Dict with parameters from toml
parms_toml = Dict()


read_parameters()


# File paths
perambulator_file(n_cnfg, i_src) = parms.perambulator_dir/"perambulator_" *
                                   "$(parms_toml["Perambulator"]["label_base"])$(i_src)_" *
                                   "$(parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = parms.mode_doublets_dir/
                             "mode_doublets_$(parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = parms.sparse_modes_dir/
                            "sparse_modes_$(parms_toml["Run name"]["name"])n$(n_cnfg)"

# Get momentum index
p_arr = read_mode_doublet_momenta(mode_doublets_file(parms.cnfg_indices[1]))
iₚ = findfirst(p -> p == parms.p, eachrow(p_arr))
if isnothing(iₚ)
    throw(DomainError("the chosen momentum 'p' is not contained in mode doublets."))
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = parms.cnfg_indices[1]

# Perambulator and mode doublets array
τ_αkβlt = allocate_perambulator()
Φ_kltiₚ = allocate_mode_doublets(mode_doublets_file(n_cnfg))

# Sparse mode arrays
sparse_modes_arrays = allocate_sparse_modes(sparse_modes_file(n_cnfg))
if parms_toml["Increased Separation"]["increase_sep"]
    N_sep_new = parms_toml["Increased Separation"]["N_sep_new"]
    N_points = prod(parms.Nₖ)÷N_sep_new^3 
    sparse_modes_arrays_new = allocate_sparse_modes(N_points=N_points)
end

correlator = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator2 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)
correlator3 = Array{ComplexF64}(undef, parms.Nₜ, parms.N_src, parms.N_cnfg)


# %%#########
# Computation
#############

for (i_cnfg, n_cnfg) in enumerate(parms.cnfg_indices)
    println("Configuration $n_cnfg")
    @time "Finished configuration $n_cnfg" begin
        @time "  Read sparse modes " begin
            read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
            if parms_toml["Increased Separation"]["increase_sep"]
                N_sep_new = parms_toml["Increased Separation"]["N_sep_new"]
                increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays,
                                     N_sep_new, n_cnfg)
            end
        end
        @time "  Read mode doublets" begin
            read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
        end
        println()

        for (i_src, t₀) in enumerate(parms.tsrc_arr[i_cnfg, :])
            println("  Source: $i_src of $(parms.N_src)")

            @time "    Read perambulator" begin
                read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
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

#= #= path = "/home/stumpa/Seafile/Dokumente/HU_Berlin,DESY/Programs/wit_-_MainzLattice/PerambulatorContractions/run_pseudoscalar_juwels_tmp0/program_files/results_(p0,0,0)"
correlator = HDF5.h5read("$path/B450r000_32modes_pseudoscalar.hdf5", "Correlator")
correlator2 = HDF5.h5read("$path/B450r000_32modes_pseudoscalar_p0.hdf5", "Correlator")
correlator3 = HDF5.h5read("$path/B450r000_32modes_pseudoscalar_sparse.hdf5", "Correlator") =#

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
# Plt.scatter!(1:Nₜ, corr2, label="For zero momentum")
Plt.scatter!(1:Nₜ, corr3, label="Position space sampling")
#= for i in 1:parms.N_cnfg
    Plt.plot!(legend=false)
    Plt.scatter!(1:Nₜ, corr_[:, 1, i][corr_[:, 1, i].>0.0], label="Using mode doublets")
    #Plt.scatter!(1:Nₜ, corr2_[:, 1, i][corr2_[:, 1, i].>0.0], label="Using full eigenvectors")

end =#
display(plot)

# Plt.savefig(p, "pseudoscalar_p1,0,0_Nsep1.pdf")


# %%
# Argument of correlaztor
denom = 16

corr_complex = vec(Stats.mean(correlator, dims=(2, 3)))
corr3_complex = vec(Stats.mean(correlator3, dims=(2, 3)))

p = Plt.plot(xlabel=L"t/a", ylabel=L"\arg(C(t))", ylims=2π./denom.*[-2.5, 2.5])
Plt.hline!([2π/denom], label=L"\pm 2\pi/%$denom, \pm 4\pi/%$denom", color=:black)
Plt.hline!([4π/denom], label=nothing, color=:black)
Plt.hline!([-2π/denom], label=nothing, color=:black)
Plt.hline!([-4π/denom], label=nothing, color=:black)
Plt.scatter!(0:Nₜ, angle.(corr_complex), label="Using mode doublets")
Plt.scatter!(0:Nₜ, angle.(corr3_complex), label="Position space sampling") =#


# %%