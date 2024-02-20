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

include("IO.jl")
include("contractions.jl")

# Add infile manually to arguments
# pushfirst!(ARGS, "-i", "run_pseudoscalar/input/pseudoscalar_16x8v1.toml")


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


# %%

function increase_separation(sparse_modes_arrays, N_sep_new, n_cnfg)
    if mod(N_sep_new, 2) != 0
        throw(ArgumentError("'N_sep_new' has to be a multiple of 2."))
    end
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Determine seperation in current sparse modes
    N_sep = (prod(parms.Nₖ)/size(x_sink_μiₓ)[2])^(1/3)
    N_sep = round(Int, N_sep)

    # Create array with indices of new sparse modes
    iₓ_arr = collect(1:size(x_sink_μiₓ)[2])
    Nₖ_sparse = round(Int, length(iₓ_arr)^(1/3))
    iₓ_arr = reshape(iₓ_arr, (Nₖ_sparse, Nₖ_sparse, Nₖ_sparse))
    iₓ_arr = permutedims(iₓ_arr, (3, 2, 1))
    division = N_sep_new÷N_sep
    if division == 0
        throw(ArgumentError("'N_sep_new' has to be bigger than the N_sep in "*
                            "sparse_modes_arrays."))
    end

    seed = parms_toml["Increased Separation"]["seed"]
    seed_cnfg = seed ⊻ n_cnfg
    rng = Random.MersenneTwister(seed_cnfg)
    
    iₓ_sink_new_arr = vec(iₓ_arr[1:division:end, 1:division:end, 1:division:end])
    iₓ_src_new_arr = Array{Int}(undef, length(iₓ_sink_new_arr), parms.Nₜ)
    for iₜ in 1:parms.Nₜ
        offset = rand(rng, 1:division, 3)
        iₓ_src_new_arr[:, iₜ] = vec(iₓ_arr[offset[1]:division:end,
                                           offset[2]:division:end,
                                           offset[3]:division:end])
    end

    # Create new sparse spaces/modes at sink
    x_sink_new_μiₓ = x_sink_μiₓ[:, iₓ_sink_new_arr]
    v_sink_new_ciₓkt = v_sink_ciₓkt[:, iₓ_sink_new_arr, :, :]

    # Create new sparse spaces/modes at sink
    x_src_new_μiₓt = Array{Int}(undef, size(x_sink_new_μiₓ)..., parms.Nₜ)
    v_src_new_ciₓkt = Array{ComplexF64}(undef, size(x_sink_new_μiₓ)..., parms.N_modes, parms.Nₜ)
    for iₜ in 1:parms.Nₜ
        x_src_new_μiₓt[:, :, iₜ] = x_src_μiₓt[:, iₓ_src_new_arr[:, iₜ], iₜ]
        v_src_new_ciₓkt[:, :, :, iₜ] = v_src_ciₓkt[:, iₓ_src_new_arr[:, iₜ], :, iₜ]
    end

    return x_sink_new_μiₓ, x_src_new_μiₓt, v_sink_new_ciₓkt, v_src_new_ciₓkt
end


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
            if parms_toml["Increased Separation"]["increase_sep"]
                N_sep_new = parms_toml["Increased Separation"]["N_sep_new"]
                sparse_modes_arrays = increase_separation(sparse_modes_arrays, N_sep_new,
                                                          n_cnfg)
            end
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
    GC.gc()
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
display(plot) =#

# Plt.savefig(p, "pseudoscalar_p1,0,0_Nsep1.pdf")


# %%