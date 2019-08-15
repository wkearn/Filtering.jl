module Filtering

using LinearAlgebra, Statistics, Optim, ProgressMeter

include("models.jl")
include("filter.jl")
include("smoother.jl")
include("em.jl")
include("resampling.jl")
include("particle.jl")
include("score_particle.jl")
#include("bare_particle.jl")
#include("gradient_particle.jl") # This may be outdated
#include("online_particle.jl")
#include("threads.jl")
include("optimizers.jl")

# TODO: These models should be moved to a FilteringModels package
include("models/ar1.jl")
include("models/smoothingspline.jl")
include("models/ss_tfi.jl")
include("models/tfi.jl")

end # module
