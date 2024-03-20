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

include("functions/allocate_arrays.jl")
include("functions/contractions.jl")
include("functions/IO.jl")
include("functions/mpi_utils.jl")
include("functions/utils.jl")
include("functions/variables.jl")

# Precompile module
include("functions/precompilation.jl")

end # module PerambulatorContractions