#######################
# This accumulates a stochastic estimate
# of the score using the algorithm of
# Nemeth et al. (2017).
export score, score_filter

struct ScoreParticleContainer <: ParticleContainer
    X # Particles
    w # weights
    α
    ℓ # Estimate of the log likelihood
    S # Score
    dg # Preallocated GradientResult
end

Base.size(ps::ScoreParticleContainer) = (size(ps.X,2),)
Base.length(ps::ScoreParticleContainer) = prod(size(ps))
Base.getindex(ps::ScoreParticleContainer,i::Int) = ps.X[:,i]
Base.setindex!(ps::ScoreParticleContainer,v,i::Int) = ps.X[:,i] = v
Base.IndexStyle(ps::ScoreParticleContainer) = IndexLinear()

StatsBase.score(ps::ScoreParticleContainer) = ps.S
StatsBase.loglikelihood(ps::ScoreParticleContainer) = ps.ℓ[end]

resample!(ps::ScoreParticleContainer,m,t,θ,y,u,sample_function=resample_stratified) = resample!(ScoreParticleContainer,ps.X,ps.w,ps.α,ps.w[:,t],m,t,θ,y,u,sample_function)

function resample!(::Type{ScoreParticleContainer},X,w,α,ξs,m,t,θ,y,u,sample_function)
    for i in eachindex(ξs)
        ξs[i] += weight_function(m,X[:,i,t],y,u,t,θ)
    end
    Σ = sum(exp,ξs)
    idx = sample_function(exp.(ξs)./Σ,length(ξs))

    X[:,:,t] .= X[:,idx,t]
    w[:,t] .= w[idx,t]
    α[:,:,t] .= α[:,idx,t]
    ξs[idx],Σ
end

propagate!(ps::ScoreParticleContainer,ξ,Σ,m,t,θ,y,u,λ) = propagate!(ScoreParticleContainer,ps.X,ps.w,ps.α,ps.ℓ,ps.S,ps.dg,ξ,Σ,m,t,θ,y,u,λ)

function propagate!(::Type{ScoreParticleContainer},X,w,α,ℓ,S,dg,ξ,Σ,m,t,θ,y,u,λ)
    for i in eachindex(ξ)
        #dq = proposal_distribution(m,X[:,i,t],y,u,t,θ)
        #rand!(dq,view(X,:,i,t+1))
        proposal_rand!(view(X,:,i,t+1),m,X[:,i,t],y,u,t,θ)

        # We will need to define this more clearly
        dg = ssm_gradient!(dg,m,X[:,i,t+1],X[:,i,t],y,u,t,θ)
        
        w[i,t+1] = w[i,t]
        # _ssm_gradient!(...) should also calculate the
        # value of the two logpdfs.
        w[i,t+1] += DiffResults.value(dg)
        #w[i,t+1] += transition_logpdf(m,X[:,i,t+1],X[:,i,t],u,t,θ)
        #w[i,t+1] += observation_logpdf(m,y,X[:,i,t+1],u,t,θ)
        #w[i] *= pdf(f(X[:,i],t,θ),Xn)
        #w[i] *= pdf(g(Xn,t,θ),y)
        w[i,t+1] -= ξ[i]
        w[i,t+1] -= proposal_logpdf(m,X[:,i,t+1],X[:,i,t],y,u,t,θ)

        # This part is unique to the score filter
        α[:,i,t+1] .= α[:,i,t]
        α[:,i,t+1] .*= λ
        α[:,i,t+1] .+= (1-λ)*S[:] .+ DiffResults.gradient(dg)        
    end
    ℓn = log(sum(exp,w[:,t+1]))
    ℓ[t+1] = ℓ[t] + ℓn - log(length(ξ)) + log(Σ)
    w[:,t+1] .-= ℓn
    S[:] .= (α[:,:,t+1]*exp.(w[:,t+1]))
end

function step(ps::ScoreParticleContainer,m::StateSpaceModel,t,θ,y,u;sample_function=resample_stratified,λ=0.95)

    # Resample

    #rs = Resampler(ps,m,y,t,θ,sample_function)

    ξs,Σ = resample!(ps,m,t,θ,y,u,sample_function)
    
    # Propagate
    
    #ps2 = ScoreParticleContainer([propose(t,x,w,ξ,α,ps.S,y,θ,m.f,m.g,m.q,m.dp,ps.dg,λ) for (x,w,ξ,α) in rs],ps.ℓ,ps.dg,sum(rs.ξ),λ)

    propagate!(ps,ξs,Σ,m,t,θ,y,u,λ)
    
    #normalize_weights!(ps2)

    ps
end

function score_filter(m::StateSpaceModel,θ,Y,N,u=fill([0.0],length(Y));sample_function=resample_stratified,verbose=false,λ=0.95)

    T = length(Y)

    if verbose
        p = Progress(T,1)
    end
    
    ps = initialize_particles(ScoreParticleContainer,m,θ,Y,N)
    
    for t in 1:T
        ps = step(ps,m,t,θ,Y[t],u[t],sample_function=sample_function,λ=λ)
        if verbose
            next!(p)
        end
    end
    
    ps    
end

function initialize_particles(::Type{ScoreParticleContainer},m::StateSpaceModel,θ,Y,N)

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

    dg = DiffResults.GradientResult(θ)

    α = Array{Float64}(undef,length(θ),N,T+1)

    for i in 1:N
        # TODO: Implement this
        dg = initial_gradient!(dg,m,X0[:,i,1],θ)

        α[:,i,1] = DiffResults.gradient(dg)
    end

    S = α[:,:,1] * exp.(w[:,1])

    ScoreParticleContainer(X0,w,α,ℓ,S,dg)
end

#=
# Does it make sense to FB smooth the score filter? 
# We can just make it identical to the particle smoother
# or we can figure out how to smooth the gradients as well

function particle_smoother(m::StateSpaceModel,θ,Y,pf::ScoreParticleContainer,B,u=fill([0.0],length(Y));sample_function=resample_stratified)
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
            # We still need u!
                wsmooth[i] += logpdf(transition_distribution(m,pf.X[:,i,t+1],u[t+1],t,θ),x[:,t+2,j])
            end
        
            x[:,t+1,j] = pf.X[:,sample_function(exp.(wsmooth)./sum(exp,wsmooth),1),t+1]
        end
    end
    x
end
=#
