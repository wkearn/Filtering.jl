export StateSpaceModel, LinearStateSpaceModel, TimeVariantLinearStateSpaceModel, GenericStateSpaceModel, ProposalStateSpaceModel, BootstrapStateSpaceModel, initial_distribution, transition_distribution, observation_distribution, proposal_distribution, weight_function,
    initial_mean,
    initial_covariance,
    state_transition_matrix,
    state_covariance,
    state_input_matrix,
    observation_matrix,
    observation_covariance,
    observation_input_matrix,
    state_noise_transformation_matrix

abstract type StateSpaceModel end

# Default definitions of the modeling interface functions
initial_rand(m::StateSpaceModel,θ,N) = rand(initial_distribution(m,θ),N)
initial_logpdf(m::StateSpaceModel,x,θ) = logpdf(initial_distribution(m,θ),x)

initial_gradient!(dg,m::StateSpaceModel,x,θ) = ForwardDiff.gradient!(dg,θ->initial_logpdf(m,x,θ),θ)

transition_rand(m::StateSpaceModel,x,u,t,θ) = rand(transition_distribution(m,x,u,t,θ))
transition_logpdf(m::StateSpaceModel,xn,x,u,t,θ) = logpdf(transition_distribution(m,x,u,t,θ),xn)

observation_rand(m::StateSpaceModel,x,u,t,θ) = rand(observation_distribution(m,x,u,t,θ))
observation_logpdf(m::StateSpaceModel,y,x,u,t,θ) = logpdf(observation_distribution(m,x,u,t,θ),y)

ssm_gradient!(dg,m::StateSpaceModel,xn,x,y,u,t,θ) = ForwardDiff.gradient!(dg,θ->transition_logpdf(m,xn,x,u,t,θ) + observation_logpdf(m,y,x,u,t,θ))

proposal_rand(m::StateSpaceModel,x,y,u,t,θ) = rand(proposal_distribution(m,x,y,u,t,θ))
proposal_rand!(xn,m::StateSpaceModel,x,y,u,t,θ) = rand!(proposal_distribution(m,x,y,u,t,θ),xn)
proposal_logpdf(m::StateSpaceModel,xn,x,y,u,t,θ) = logpdf(proposal_distribution(m,x,y,u,t,θ),xn)

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

function initial_distribution(m::LinearStateSpaceModel,θ0)
    μ0 = m.μ(θ0)
    Σ0 = m.Σ0(θ0)

    MvNormal(μ0,Σ0)
end

function transition_distribution(m::LinearStateSpaceModel,x,u,t,θ0)
    F = m.F(θ0)
    G = m.G(θ0)
    Q = m.Q(θ0)
    Θ = m.Θ(θ0)

    MvNormal(F*x+G*u,Θ*Q*Θ')
end

function observation_distribution(m::LinearStateSpaceModel,x,u,t,θ0)
    H = m.H(θ0)
    Γ = m.Γ(θ0)
    R = m.R(θ0)

    MvNormal(H*x + Γ*u,R)
end

function proposal_distribution(m::LinearStateSpaceModel,x,y,u,t,θ)
    transition_distribution(m,x,u,t,θ)
end

function weight_function(m::LinearStateSpaceModel,x,y,u,t,θ)
    0.0
end

### Linear model interface

initial_mean(m::LinearStateSpaceModel,θ) = m.μ(θ)
initial_covariance(m::LinearStateSpaceModel,θ) = m.Σ0(θ)
state_transition_matrix(m::LinearStateSpaceModel,θ) = m.F(θ)
state_covariance(m::LinearStateSpaceModel,θ) = m.Q(θ)
state_input_matrix(m::LinearStateSpaceModel,θ) = m.G(θ)
observation_matrix(m::LinearStateSpaceModel,θ) = m.H(θ)
observation_covariance(m::LinearStateSpaceModel,θ) = m.R(θ)
observation_input_matrix(m::LinearStateSpaceModel,θ) = m.Γ(θ)
state_noise_transformation_matrix(m::LinearStateSpaceModel,θ) = m.Θ(θ)

struct LinearStateSpaceGradient
    dF
    dQ
    dH
    dR
    dΘ
    dμ
    dΣ0
end

function simulate(m::LinearStateSpaceModel,θ,N,u=fill([0.0],N))
    μ0 = initial_distribution(m,θ)
    X = fill(rand(μ0),N+1)
    Y = fill(rand(observation_distribution(m,X[1],u[1],0,θ)),N)
    for t in 1:N
        X[t+1] = rand(transition_distribution(m,X[t],u[t],t,θ))
        Y[t] = rand(observation_distribution(m,X[t+1],u[t],t,θ))
    end
    X,Y
end

#=
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
=#

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

initial_distribution(m::ProposalStateSpaceModel,θ) = m.μ0(θ)

transition_distribution(m::ProposalStateSpaceModel,x,u,t,θ) = m.f(x,u,t,θ)

observation_distribution(m::ProposalStateSpaceModel,x,u,t,θ) = m.g(x,u,t,θ)

proposal_distribution(m::ProposalStateSpaceModel,x,y,u,t,θ) = m.q(x,y,u,t,θ)

weight_function(m::ProposalStateSpaceModel,x,y,u,t,θ) = m.ξ(x,y,u,t,θ)

function DP(f,g)
    function dp(dh,x,xn,y,t,θp)
        ForwardDiff.gradient!(dh,θ->logpdf(g(xn,t,θ),y) + logpdf(f(x,t,θ),xn),θp)
    end
end

function BootstrapStateSpaceModel(μ0,f,g)
    q(x,y,u,t,θ) = f(x,u,t,θ)
    ξ(x,y,u,t,θ) = 0.0 # We do this in logarithms now
    ProposalStateSpaceModel(μ0,f,g,q,ξ)
end

function simulate(m::ProposalStateSpaceModel,θ,N,u=fill([0.0],N))
    X = fill(rand(m.μ0(θ)),N+1)
    Y = fill(rand(m.g(X[1],u[1],1,θ)),N)
    for t in 1:N
        X[t+1] =  rand(m.f(X[t],u[t],t,θ))
        Y[t] = rand(m.g(X[t+1],u[t],t,θ))
    end
    X,Y
end

(m::ProposalStateSpaceModel)(θ,N) = simulate(m,θ,N)
