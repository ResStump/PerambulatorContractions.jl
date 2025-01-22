module PerambulatorContractions

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import MPI
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import ArgParse as AP
import Combinatorics as Comb
import Dates
import PrecompileTools

# Disable multithreading in BLAS operations
LA.BLAS.set_num_threads(1)

include("functions/allocate_arrays.jl")
include("functions/contractions_meson.jl")
include("functions/contractions_DD.jl")
include("functions/contractions_dad.jl")
include("functions/contractions_DD-dad_mixed.jl")
include("functions/IO.jl")
include("functions/mpi_utils.jl")
include("functions/utils.jl")
include("functions/variables.jl")

# Precompile module
include("functions/precompilation.jl")

end # module PerambulatorContractions