using DiffResults, ForwardDiff, Distributions, StatsBase, StatsFuns

export ScoreParticleContainer

abstract type ParticleContainer <: AbstractArray{Float64,1}
end

struct ScoreParticleContainer <: ParticleContainer
    X # Particles
    w # weights
    α # Mean of the score vector
    ℓ # Estimate of the log likelihood
    S # Score vector
    dg # Preallocated GradientResult
end

Base.size(ps::ScoreParticleContainer) = (size(ps.X,2),)
Base.length(ps::ScoreParticleContainer) = prod(size(ps))
Base.getindex(ps::ScoreParticleContainer,i::Int) = ps.X[:,i]
Base.setindex!(ps::ScoreParticleContainer,v,i::Int) = ps.X[:,i] = v
Base.IndexStyle(ps::ScoreParticleContainer) = IndexLinear()

resample!(ps,m,y,t,θ,sample_function=resample_stratified) = resample!(ps.X,ps.w,ps.α,ps.w[:],m.ξ,y,t,θ,length(ps.w),sample_function)

function resample!(X,w,α,ξs,ξ,y,t,θ,N,sample_function)
    for i in 1:N
        ξs[i] += ξ(X[:,i],y,t,θ)
    end
    Σ = logsumexp(ξs)
    idx = sample_function(exp.(ξs .- Σ),N)

    X[:,:] .= X[:,idx]
    w[:] .= w[idx]
    α[:,:] .= α[:,idx]
    ξs[idx],Σ
end

propagate!(ps,ξ,Σ,m,y,t,θ,λ) = propagate!(ps.X,ps.w,ps.α,ps.dg,ps.ℓ,ps.S,ξ,Σ,m.f,m.g,m.q,m.dp,y,t,θ,length(ps.w),λ)

function propagate!(X,w,α,dg,ℓ,S,ξ,Σ,f,g,q,dp,y,t,θ,N,λ)
    for i in 1:N
        dq = q(X[:,i],y,t,θ)
        Xn = rand(dq)

        dg = dp(dg,X[:,i],Xn,y,t,θ)   
        
        w[i] += DiffResults.value(dg)
        #w[i] *= pdf(f(X[:,i],t,θ),Xn)
        #w[i] *= pdf(g(Xn,t,θ),y)
        w[i] -= ξ[i]
        w[i] -= logpdf(dq,Xn)

        
        α[:,i] *= λ
        α[:,i] .+= (1-λ)*S[:] .+ DiffResults.gradient(dg)
        X[:,i] .= Xn[:]
    end
    ℓn = logsumexp(w)
    ℓ[t+1] = ℓ[t] + ℓn - log(N) + Σ
    w .-= ℓn
    S[:] .= (α*exp.(w))[:] 
end

function step(ps::ScoreParticleContainer,m::ProposalStateSpaceModel,y,t,θ,λ=0.95;Nthreshold=0.5,sample_function=resample_stratified)

    # Resample

    #rs = Resampler(ps,m,y,t,θ,sample_function)

    ξs,Σ = resample!(ps,m,y,t,θ,sample_function)
    
    # Propagate
    
    #ps2 = ScoreParticleContainer([propose(t,x,w,ξ,α,ps.S,y,θ,m.f,m.g,m.q,m.dp,ps.dg,λ) for (x,w,ξ,α) in rs],ps.ℓ,ps.dg,sum(rs.ξ),λ)

    propagate!(ps,ξs,Σ,m,y,t,θ,λ)
    
    #normalize_weights!(ps2)

    ps
end

function score_filter(m::ProposalStateSpaceModel,Y,θ,N;λ=0.95,Nthreshold=0.5,sample_function=resample_stratified,verbose=false)

    T = length(Y)

    μ0 = m.μ0
    f = m.f
    g = m.g

    if verbose
        p = Progress(T,1)
    end
    
    ps = initialize_particles(m,θ,Y,N,λ)
    
    for t in 1:T
        ps = step(ps,m,Y[t],t,θ,λ,Nthreshold=Nthreshold,sample_function=sample_function)
        if verbose            
            next!(p)
        end
    end
    
    ps    
end

function initialize_particles(μ0,f,g,θ,Y,N,T,λ;save_filter=false)

    # We should probably think of μ0 as a distribution for p(x_0)
    # not one for p(x_1)
    # Initialization gives X0, w = 1/N
    #x0 = rand(μ0(θ),N)

    X0 = rand(μ0(θ),N)
    # This used to be: cat(rand(μ0(θ),N)...,dims=3)
    # I think this was to make it work for matrix-valued random variables
        
    w = log.(ones(N)/N)
    ℓ = zeros(T+1)
    #w = [pdf(g(X0[:,end,i],1,θ),y)/N for i in 1:N]
    #ℓ = sum(w)
    #w ./= ℓ

    dg = DiffResults.GradientResult(θ)

    α = Array{Float64}(undef,length(θ),N)

    for i in 1:N        
        dg = ForwardDiff.gradient!(dg,θ->logpdf(μ0(θ),X0[:,i]),θ)
    
        α[:,i] = DiffResults.gradient(dg)
    end

    S = α*exp.(w)
    
    ScoreParticleContainer(X0,w,α,ℓ,S,dg)
end
initialize_particles(m::ProposalStateSpaceModel,θ,Y,N,λ) = initialize_particles(m.μ0,m.f,m.g,θ,Y,N,length(Y),λ)
