# Stochastic gradient optimizers

abstract type Optimizer end

function training_step!(opt,θ,G)
    optimizer_step!(opt,G)
    θ + opt.Δ
end

"""
Maximize likelihood with stochastic gradient

- dℓ(θ) returns loglikelihood and gradient
"""
function train(dℓ,θ0,opt;max_iter=1000, verbose=false, progress=false)
    θ = zeros(length(θ0))
    θ .= θ0[:]
    ℓ = zeros(max_iter)
    if progress
        p = Progress(max_iter,1)
    end
    for i in 1:max_iter
        ℓ[i],G = dℓ(θ)
        θ .= training_step!(opt,θ,G)
        if progress
            next!(p)#; showvalues = [(:iteration, i),(:likelihood,ℓ[i])])
        end
        if verbose
            println("Step: ",rpad(i,5,' ')," | Likelihood: ", ℓ[i])
        end
    end
    θ,ℓ
end

mutable struct SGD <: Optimizer
    η
    δ
    Δ::Array{Float64,1}
end

SGD(n::Int,η=0.001,δ=1.0) = SGD(η,δ,zeros(n))

function optimizer_step!(opt::SGD,G)
    opt.Δ .= opt.η*G
    opt.η = opt.η*opt.δ
    opt
end

mutable struct Momentum <: Optimizer
    η
    ρ
    δ
    Δ::Array{Float64,1}
end

Momentum(n::Int,η=0.001,ρ=0.9,δ=1.0) = Momentum(η,ρ,δ,zeros(n))

function optimizer_step!(opt::Momentum,G)
    opt.Δ = opt.ρ*opt.Δ + opt.η*G
    opt.η = opt.η*opt.δ
    opt
end

mutable struct Adam <: Optimizer
    η
    δ
    β1
    β2
    βp
    m
    v
    Δ::Array{Float64,1}
end

Adam(n::Int,η=0.001,δ=1.0,β1=0.9,β2=0.999) = Adam(η,δ,β1,β2,[β1;β2],zeros(n),zeros(n),zeros(n))

function optimizer_step!(opt::Adam,G)
    opt.m .= opt.β1*opt.m + (1-opt.β1)*G
    opt.v .= opt.β2*opt.v + (1-opt.β2)*(G.^2)
    opt.Δ .= opt.η*(opt.m/(1-opt.βp[1]))./(sqrt.(opt.v/(1-opt.βp[2])).+1e-8)
    opt.βp .*= [opt.β1,opt.β2]
    opt.η = opt.δ*opt.η
    opt
end


