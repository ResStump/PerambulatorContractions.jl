import Startup
import Test:@test, @testset

include("../src/IO.jl")

# Set global parameters
#######################

# Add infile manually to arguments
pushfirst!(ARGS, "-i", "test/16x8v1_data/pseudoscalar_16x8v1.toml")

# Read parameters from infile
parms, parms_toml = read_parameters()


# Run tests
include("IO_tests.jl")
include("pseudoscalar_tests.jl")
