import TensorOperations as TO
import Startup
import Test:@test, @testset

# Include modules
include("../src/IO.jl")
include("../src/allocate_arrays.jl")
include("../src/utils.jl")
include("../src/contractions.jl")


# Run tests
include("IO_tests.jl")
include("utils_tests.jl")
include("pseudoscalar_tests.jl")
