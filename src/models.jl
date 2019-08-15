using Distributions

export StateSpaceModel, LinearStateSpaceModel, TimeVariantLinearStateSpaceModel, GenericStateSpaceModel, ProposalStateSpaceModel, BootstrapStateSpaceModel

abstract type StateSpaceModel end

"""
A linear, time-invariant state space model

A degenerate process noise is allowed by 
using Θ ≠ I, though correlated process
and measurement noise are not allowed.

To interface your own state space model
with the filtering/smoothing and parameter
estimation code, provide a function

```
buildFunction(θ)::LinearStateSpaceModel
```

where θ is a vector of parameters.
"""
struct LinearStateSpaceModel <: StateSpaceModel
    F
    Q
    G
    H
    R
    Γ
    Θ
    μ
    Σ0
end

struct LinearStateSpaceGradient
    dF
    dQ
    dH
    dR
    dΘ
    dμ
    dΣ0
end

function simulate(m::LinearStateSpaceModel,N,u=fill([0.0],N))
    d = size(m.μ,1)
    n = size(m.H*m.μ,1)
    X = rand(MvNormal(m.μ,m.Σ0),N+1)
    Y = rand(MvNormal(m.H*m.μ,m.R),N+1)
    for i in 2:N+1
        X[:,i] = m.F*X[:,i-1] + m.G*u[i-1] + rand(MvNormal(zeros(d),m.Q))
        Y[:,i] = m.H*X[:,i] .+ m.Γ*u[i-1] .+ rand(MvNormal(zeros(n),m.R))
    end
    X,Y
end

"""
A linear, time-variant state space model

Each of the model components (with the
exception of the initial state and 
covariance estimates) is a vector of 
matrices. F[i] gives the matrix F at 
the `i`th time step.

A degenerate process noise is allowed by 
using Θ ≠ I, though correlated process
and measurement noise are not allowed.

To interface your own state space model
with the filtering/smoothing and parameter
estimation code, provide a function

```
buildFunction(θ)::LinearStateSpaceModel
```

where θ is a vector of parameters.
"""
struct TimeVariantLinearStateSpaceModel <: StateSpaceModel
    F
    Q
    G
    H
    R
    Γ
    Θ
    μ
    Σ0
end

function simulate(m::TimeVariantLinearStateSpaceModel,N,u=fill([0.0],N))
    d = size(m.μ,1)
    n = size(m.H[1]*m.μ,1)
    X = rand(MvNormal(m.μ,m.Σ0),N+1)
    Y = rand(MvNormal(m.H[1]*m.μ,m.R[1]),N+1)
    for i in 2:N+1
        X[:,i] = m.F[i-1]*X[:,i-1] + m.G[i-1]*u[i-1] + rand(MvNormal(zeros(d),m.Q[i-1]))
        Y[:,i] = m.H[i-1]*X[:,i] + m.Γ[i-1]*u[i-1] + rand(MvNormal(zeros(n),m.R[i-1]))
    end
    X,Y
end

"""
A generic state space model for use with particle filters

Three functions are needed:

- `μ0(θ=θ0) :: Distribution`
- `f(x,θ=θ0) :: Distribution`
- `g(x,θ=θ0) :: Distribution`

`μ0(θ)` is the initial distribution over hidden states
i.e. `x = rand(μ0(θ))` is a sample from that initial distribution

`f(x,θ)` is the transition probability distribution to move from
x to a new state. `xn = rand(f(x,θ))` propagates a particle forward. 
`pdf(f(x,θ),xn)` evaluates the density of `f` at the new hidden state.

`g(x,θ)` is the observation probability distribution. `pdf(g(x,θ),y)`
evaluates the density of `g` at the observation `y`

If you don't want to make use of the maximum likelihood estimation tools, you can provide one-argument versions of the functions. You would then use the `particle_filter` method without the parameter.
"""
struct GenericStateSpaceModel <: StateSpaceModel
    μ0
    f
    g
end

"""
A state space model with a proposal distribution for auxiliary filtering

`μ0`, `f` and `g` resemble those for a GenericStateSpaceModel

`q(x,y,t,θ)` creates a proposal distribution at current state x, 
next observation y, time t and parameter value θ.

`ξ(x,y,t,θ)` gives the value of the weight multiplier so that particle (x,w)
is resampled with probability `w*ξ(x,y,t,θ)`.

"""
struct ProposalStateSpaceModel <: StateSpaceModel
    μ0
    f
    g
    q
    ξ
    dp
    ProposalStateSpaceModel(μ0,f,g,q,ξ,dp=DP(f,g)) = new(μ0,f,g,q,ξ,dp)
end

function DP(f,g)
    function dp(dh,x,xn,y,t,θp)
        ForwardDiff.gradient!(dh,θ->logpdf(g(xn,t,θ),y) + logpdf(f(x,t,θ),xn),θp)
    end
end

function BootstrapStateSpaceModel(μ0,f,g)
    q(x,y,t,θ) = f(x,t,θ)
    ξ(x,y,t,θ) = 0.0 # We do this in logarithms now
    ProposalStateSpaceModel(μ0,f,g,q,ξ)
end

function simulate(m::ProposalStateSpaceModel,θ,N)
    X = fill(rand(m.μ0(θ)),N+1)
    Y = fill(rand(m.g(X[1],1,θ)),N)
    for t in 1:N
        X[t+1] =  rand(m.f(X[t],t,θ))
        Y[t] = rand(m.g(X[t+1],t,θ))
    end
    X,Y
end

(m::ProposalStateSpaceModel)(θ,N) = simulate(m,θ,N)