#######################
# Filtering distribution
# This does not save trajectories, but saves weights and particles at each time step
# Thus each corresponds to the filtering distribution.
export particle_filter, particle_smoother

abstract type ParticleContainer <: AbstractArray{Float64,1}
end

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

StatsBase.loglikelihood(ps::FilteringParticleContainer) = ps.ℓ[end]

resample!(ps::FilteringParticleContainer,m,t,θ,y,u,sample_function=resample_stratified) = resample!(FilteringParticleContainer,ps.X,ps.w,ps.w[:,t],m,t,θ,y,u,sample_function)

function resample!(::Type{FilteringParticleContainer},X,w,ξs,m,t,θ,y,u,sample_function)
    for i in eachindex(ξs)
        ξs[i] += weight_function(m,X[:,i,t],y,u,t,θ)
    end
    Σ = sum(exp,ξs)
    idx = sample_function(exp.(ξs)./Σ,length(ξs))

    X[:,:,t] .= X[:,idx,t]
    w[:,t] .= w[idx,t]
    ξs[idx],Σ
end

propagate!(ps::FilteringParticleContainer,ξ,Σ,m,t,θ,y,u) = propagate!(FilteringParticleContainer,ps.X,ps.w,ps.ℓ,ξ,Σ,m,t,θ,y,u)

function propagate!(::Type{FilteringParticleContainer},X,w,ℓ,ξ,Σ,m,t,θ,y,u)
    for i in eachindex(ξ)
        #dq = proposal_distribution(m,X[:,i,t],y,u,t,θ)
        #rand!(dq,view(X,:,i,t+1))
        proposal_rand!(view(X,:,i,t+1),m,X[:,i,t],y,u,t,θ)

        w[i,t+1] = w[i,t]
        w[i,t+1] += transition_logpdf(m,X[:,i,t+1],X[:,i,t],u,t,θ)
        w[i,t+1] += observation_logpdf(m,y,X[:,i,t+1],u,t,θ)
        #w[i] *= pdf(f(X[:,i],t,θ),Xn)
        #w[i] *= pdf(g(Xn,t,θ),y)
        w[i,t+1] -= ξ[i]
        w[i,t+1] -= proposal_logpdf(m,X[:,i,t+1],X[:,i,t],y,u,t,θ)

        
    end
    ℓn = log(sum(exp,w[:,t+1]))
    ℓ[t+1] = ℓ[t] + ℓn - log(length(ξ)) + log(Σ)
    w[:,t+1] .-= ℓn
end

function step(ps::FilteringParticleContainer,m::StateSpaceModel,t,θ,y,u;sample_function=resample_stratified)

    # Resample

    #rs = Resampler(ps,m,y,t,θ,sample_function)

    ξs,Σ = resample!(ps,m,t,θ,y,u,sample_function)
    
    # Propagate
    
    #ps2 = ScoreParticleContainer([propose(t,x,w,ξ,α,ps.S,y,θ,m.f,m.g,m.q,m.dp,ps.dg,λ) for (x,w,ξ,α) in rs],ps.ℓ,ps.dg,sum(rs.ξ),λ)

    propagate!(ps,ξs,Σ,m,t,θ,y,u)
    
    #normalize_weights!(ps2)

    ps
end

function particle_filter(m::StateSpaceModel,θ,Y,N,u=fill([0.0],length(Y));sample_function=resample_stratified,verbose=false)

    T = length(Y)

    if verbose
        p = Progress(T,1)
    end
    
    ps = initialize_particles(FilteringParticleContainer,m,θ,Y,N)
    
    for t in 1:T
        ps = step(ps,m,t,θ,Y[t],u[t],sample_function=sample_function)
        if verbose
            next!(p)
        end
    end
    
    ps    
end

function initialize_particles(::Type{FilteringParticleContainer},m::StateSpaceModel,θ,Y,N)

    # We should probably think of μ0 as a distribution for p(x_0)
    # not one for p(x_1)
    # Initialization gives X0, w = 1/N
    #x0 = rand(μ0(θ),N)
    T = length(Y)

    x0 = initial_rand(m,θ,N)
    X0 = zeros(size(x0,1),N,T+1)
    X0[:,:,1] = x0[:,:]
        
    w = log.(ones(N,T+1)/N)
    ℓ = zeros(T+1)
    #w = [pdf(g(X0[:,end,i],1,θ),y)/N for i in 1:N]
    #ℓ = sum(w)
    #w ./= ℓ

    FilteringParticleContainer(X0,w,ℓ)
end

function particle_smoother(m::StateSpaceModel,θ,Y,pf::FilteringParticleContainer,B,u=fill([0.0],length(Y));sample_function=resample_stratified)
    D = size(pf.X,1)
    N = size(pf.X,2)
    T = size(pf.X,3)-1
    
    x = zeros(D,T+1,B)
    for i in 1:B
        x[:,end,i] = pf.X[:,sample_function(exp.(pf.w[:,end])./sum(exp,pf.w[:,end]),1),end]
    end
   
    for t in T-1:-1:0
        wsmooth = pf.w[:,t+1]
        for j in 1:B
            for i in 1:N
            # This is quite slow
                wsmooth[i] += transition_logpdf(m,x[:,t+2,j],pf.X[:,i,t+1],u[t+1],t,θ)
            end
        
            x[:,t+1,j] = pf.X[:,sample_function(exp.(wsmooth)./sum(exp,wsmooth),1),t+1]
        end
    end
    x
end
