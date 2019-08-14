using DiffResults, ForwardDiff, Distributions, StatsBase

export BareParticleContainer

struct BareParticleContainer <: ParticleContainer
    X # Particles
    w # weights
    ℓ # Estimate of the log likelihood
end

Base.size(ps::BareParticleContainer) = (size(ps.X,2),)
Base.length(ps::BareParticleContainer) = prod(size(ps))
Base.getindex(ps::BareParticleContainer,i::Int) = ps.X[:,i]
Base.setindex!(ps::BareParticleContainer,v,i::Int) = ps.X[:,i] = v
Base.IndexStyle(ps::BareParticleContainer) = IndexLinear()

resample!(ps::BareParticleContainer,m,y,t,θ,sample_function=resample_stratified) = resample!(BareParticleContainer,ps.X,ps.w,ps.w[:],m.ξ,y,t,θ,length(ps.w),sample_function)

function resample!(::Type{BareParticleContainer},X,w,ξs,ξ,y,t,θ,N,sample_function)
    for i in 1:N
        ξs[i] += ξ(X[:,i,t],y,t,θ)
    end
    Σ = sum(exp,ξs)
    idx = sample_function(exp.(ξs)./Σ,N)

    X[:,:,1:t] .= X[:,idx,1:t]
    w[:] .= w[idx]
    ξs[idx],Σ
end

propagate!(ps::BareParticleContainer,ξ,Σ,m,y,t,θ,λ) = propagate!(BareParticleContainer,ps.X,ps.w,ps.ℓ,ξ,Σ,m.f,m.g,m.q,m.dp,y,t,θ,length(ps.w),λ)

function propagate!(::Type{BareParticleContainer},X,w,ℓ,ξ,Σ,f,g,q,dp,y,t,θ,N,λ)
    for i in 1:N
        dq = q(X[:,i,t],y,t,θ)
        X[:,i,t+1] = rand(dq)

        w[i] += logpdf(f(X[:,i,t],t,θ),X[:,i,t+1])
        w[i] += logpdf(g(X[:,i,t+1],t,θ),y)
        #w[i] *= pdf(f(X[:,i],t,θ),Xn)
        #w[i] *= pdf(g(Xn,t,θ),y)
        w[i] -= ξ[i]
        w[i] -= logpdf(dq,X[:,i,t+1])

        
    end
    ℓn = log(sum(exp,w))
    ℓ[t+1] = ℓ[t] + ℓn - log(N) + log(Σ)
    w .-= ℓn
end

function step(ps::BareParticleContainer,m::ProposalStateSpaceModel,y,t,θ,λ=0.95;Nthreshold=0.5,sample_function=resample_stratified)

    # Resample

    #rs = Resampler(ps,m,y,t,θ,sample_function)

    ξs,Σ = resample!(ps,m,y,t,θ,sample_function)
    
    # Propagate
    
    #ps2 = ScoreParticleContainer([propose(t,x,w,ξ,α,ps.S,y,θ,m.f,m.g,m.q,m.dp,ps.dg,λ) for (x,w,ξ,α) in rs],ps.ℓ,ps.dg,sum(rs.ξ),λ)

    propagate!(ps,ξs,Σ,m,y,t,θ,λ)
    
    #normalize_weights!(ps2)

    ps
end

function bare_filter(m::ProposalStateSpaceModel,Y,θ,N;λ=0.95,Nthreshold=0.5,sample_function=resample_stratified,verbose=false)

    T = length(Y)

    μ0 = m.μ0
    f = m.f
    g = m.g

    ps = initialize_particles(BareParticleContainer,m,θ,Y,N,λ)
    
    for t in 1:T
        if verbose
            println("Step: ",t)
            println("ESS: ",1 ./ sum(abs2,exp.(ps.w)))
        end
        ps = step(ps,m,Y[t],t,θ,λ,Nthreshold=Nthreshold,sample_function=sample_function)
    end
    
    ps    
end

function initialize_particles(::Type{BareParticleContainer},μ0,f,g,θ,Y,N,T,λ;save_filter=false)

    # We should probably think of μ0 as a distribution for p(x_0)
    # not one for p(x_1)
    # Initialization gives X0, w = 1/N
    #x0 = rand(μ0(θ),N)

    x0 = rand(μ0(θ),N)
    X0 = zeros(size(x0,1),N,T+1)
    X0[:,:,1] = x0[:,:]
        
    w = log.(ones(N)/N)
    ℓ = zeros(T+1)
    #w = [pdf(g(X0[:,end,i],1,θ),y)/N for i in 1:N]
    #ℓ = sum(w)
    #w ./= ℓ

    BareParticleContainer(X0,w,ℓ)
end
initialize_particles(::Type{BareParticleContainer},m::ProposalStateSpaceModel,θ,Y,N,λ) = initialize_particles(BareParticleContainer,m.μ0,m.f,m.g,θ,Y,N,length(Y),λ)

#######################
# Filtering distribution
# This does not save trajectories, but saves weights and particles at each time step
# Thus each corresponds to the filtering distribution.

struct FilteringParticleContainer <: ParticleContainer
    X # Particles
    w # weights
    ℓ # Estimate of the log likelihood
end

Base.size(ps::FilteringParticleContainer) = (size(ps.X,2),)
Base.length(ps::FilteringParticleContainer) = prod(size(ps))
Base.getindex(ps::FilteringParticleContainer,i::Int) = ps.X[:,i]
Base.setindex!(ps::FilteringParticleContainer,v,i::Int) = ps.X[:,i] = v
Base.IndexStyle(ps::FilteringParticleContainer) = IndexLinear()

resample!(ps::FilteringParticleContainer,m,y,t,θ,sample_function=resample_stratified) = resample!(FilteringParticleContainer,ps.X,ps.w,ps.w[:,t],m.ξ,y,t,θ,size(ps.w,1),sample_function)

function resample!(::Type{FilteringParticleContainer},X,w,ξs,ξ,y,t,θ,N,sample_function)
    for i in 1:N
        ξs[i] += ξ(X[:,i,t],y,t,θ)
    end
    Σ = sum(exp,ξs)
    idx = sample_function(exp.(ξs)./Σ,N)

    X[:,:,t] .= X[:,idx,t]
    w[:,t] .= w[idx,t]
    ξs[idx],Σ
end

propagate!(ps::FilteringParticleContainer,ξ,Σ,m,y,t,θ,λ) = propagate!(FilteringParticleContainer,ps.X,ps.w,ps.ℓ,ξ,Σ,m.f,m.g,m.q,m.dp,y,t,θ,size(ps.w,1),λ)

function propagate!(::Type{FilteringParticleContainer},X,w,ℓ,ξ,Σ,f,g,q,dp,y,t,θ,N,λ)
    for i in 1:N
        dq = q(X[:,i,t],y,t,θ)
        X[:,i,t+1] = rand(dq)

        w[i,t+1] = w[i,t]
        w[i,t+1] += logpdf(f(X[:,i,t],t,θ),X[:,i,t+1])
        w[i,t+1] += logpdf(g(X[:,i,t+1],t,θ),y)
        #w[i] *= pdf(f(X[:,i],t,θ),Xn)
        #w[i] *= pdf(g(Xn,t,θ),y)
        w[i,t+1] -= ξ[i]
        w[i,t+1] -= logpdf(dq,X[:,i,t+1])

        
    end
    ℓn = log(sum(exp,w[:,t+1]))
    ℓ[t+1] = ℓ[t] + ℓn - log(N) + log(Σ)
    w[:,t+1] .-= ℓn
end

function step(ps::FilteringParticleContainer,m::ProposalStateSpaceModel,y,t,θ,λ=0.95;Nthreshold=0.5,sample_function=resample_stratified)

    # Resample

    #rs = Resampler(ps,m,y,t,θ,sample_function)

    ξs,Σ = resample!(ps,m,y,t,θ,sample_function)
    
    # Propagate
    
    #ps2 = ScoreParticleContainer([propose(t,x,w,ξ,α,ps.S,y,θ,m.f,m.g,m.q,m.dp,ps.dg,λ) for (x,w,ξ,α) in rs],ps.ℓ,ps.dg,sum(rs.ξ),λ)

    propagate!(ps,ξs,Σ,m,y,t,θ,λ)
    
    #normalize_weights!(ps2)

    ps
end

function filtering_filter(m::ProposalStateSpaceModel,Y,θ,N;λ=0.95,Nthreshold=0.5,sample_function=resample_stratified,verbose=false)

    T = length(Y)

    μ0 = m.μ0
    f = m.f
    g = m.g

    if verbose
        p = Progress(T,1)
    end
    
    ps = initialize_particles(FilteringParticleContainer,m,θ,Y,N,λ)
    
    for t in 1:T
        ps = step(ps,m,Y[t],t,θ,λ,Nthreshold=Nthreshold,sample_function=sample_function)
        if verbose
            next!(p)
        end
    end
    
    ps    
end

function initialize_particles(::Type{FilteringParticleContainer},μ0,f,g,θ,Y,N,T,λ;save_filter=false)

    # We should probably think of μ0 as a distribution for p(x_0)
    # not one for p(x_1)
    # Initialization gives X0, w = 1/N
    #x0 = rand(μ0(θ),N)

    x0 = rand(μ0(θ),N)
    X0 = zeros(size(x0,1),N,T+1)
    X0[:,:,1] = x0[:,:]
        
    w = log.(ones(N,T+1)/N)
    ℓ = zeros(T+1)
    #w = [pdf(g(X0[:,end,i],1,θ),y)/N for i in 1:N]
    #ℓ = sum(w)
    #w ./= ℓ

    FilteringParticleContainer(X0,w,ℓ)
end
initialize_particles(::Type{FilteringParticleContainer},m::ProposalStateSpaceModel,θ,Y,N,λ) = initialize_particles(FilteringParticleContainer,m.μ0,m.f,m.g,θ,Y,N,length(Y),λ)
