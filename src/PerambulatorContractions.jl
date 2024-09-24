module PerambulatorContractions

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import MPI
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import Combinatorics as Comb
import Dates
import PrecompileTools

include("functions/allocate_arrays.jl")
include("functions/contractions_meson.jl")
include("functions/contractions_DD.jl")
include("functions/contractions_dad.jl")
include("functions/IO.jl")
include("functions/mpi_utils.jl")
include("functions/utils.jl")
include("functions/variables.jl")

# Precompile module
include("functions/precompilation.jl")

end # module PerambulatorContractions