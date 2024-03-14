module PerambulatorContractions

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import MPI
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import PrecompileTools

export allocate_perambulator, allocate_mode_doublets, allocate_sparse_modes
export pseudoscalar_contraction!, pseudoscalar_contraction_p0!
export pseudoscalar_sparse_contraction!
export read_parameters, read_perambulator!, read_perambulator
export read_mode_doublet_momenta, read_mode_doublets!, read_mode_doublets
export read_sparse_modes!, read_sparse_modes, write_correlator
export is_my_cnfg, broadcast_correlators!, send_correlator_to_root!
export increase_separation!, increase_separation

include("functions/allocate_arrays.jl")
include("functions/contractions.jl")
include("functions/IO.jl")
include("functions/mpi_utils.jl")
include("functions/utils.jl")

# Precompile module
include("functions/precompilation.jl")

end # module PerambulatorContractions