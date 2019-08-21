export elbo, em

function elbo(ks::KalmanSmoother,z,θ,build)
    T = length(z)
    
    mθ = build(θ)
    
    μ = mθ.μ
    Σ0 = mθ.Σ0
    F = mθ.F
    Q = mθ.Q
    Θ = mθ.Θ
    Qp = Θ*Q*Θ'
    H = mθ.H
    R = mθ.R
    
    S11 = ks.S11
    S00 = ks.S00
    S10 = ks.S10

    Syy = ks.Syy

    Sμ = ks.P[1] + (ks.x[1]-μ)*(ks.x[1]-μ)'
    
    l1 = logdet(Σ0) + tr(Σ0\Sμ)
    l2 = T * logdet(Q) + tr(pinv(Qp)*(S11 - F*S10' - S10*F' + F*S00*F'))
    l3 = T * logdet(R) + tr(pinv(R)*Syy)
    
    l1 + l2 + l3
end

function elbo(m::LinearStateSpaceModel,z,θ,build)
    kf = kalman_filter(m,z)
    ks = kalman_smoother(kf,m)
    
    elbo(ks,z,θ,build)
end

# This is not ready for prime time
function elbo_gradient(ks::KalmanSmoother,z,θ)
    T = length(z)
    
    mθ = SSTFI(θ)
    μ = mθ.μ
    Σ0 = mθ.Σ0
    F = mθ.F
    Q = mθ.Q
    Θ = mθ.Θ
    Qp = Θ*Q*Θ'
    H = mθ.H
    R = mθ.R
    
    dm = Filtering._gradient(θ)
    dF = dm.dF
    dQ = dm.dQ
    dH = dm.dH
    dR = dm.dR
    dΘ = dm.dΘ
    dμ = dm.dμ
    dΣ0 = dm.dΣ0
    
    S11 = ks.S11
    S00 = ks.S00
    S10 = ks.S10
    
    Syy = ks.Syy
    
    Sμ = ks.P[1] + (ks.x[1]-μ)*(ks.x[1]-μ)'
    
    # l1
    l1 = logdet(Σ0) + tr(Σ0\Sμ)
    dl1 = zeros(length(θ0))
    
    # l2
    Θp = pinv(Θ)
    Sf =(S11 - F*S10' - S10*F' + F*S00*F')
    l2 = T*logdet(Q) + tr(pinv(Qp)*Sf)
    dl2 = [T*tr(Q\dQ[:,:,i]) +
        tr(-Θp'*(Q\dQ[:,:,i])*(Q\Θp)*Sf + 
            pinv(Qp)*(-dF[:,:,i]*S10' - S10*dF[:,:,i]' + dF[:,:,i]*S00*F' + F*S00*dF[:,:,i]')) 
        for i in 1:length(θ)]
    
    # l3
    l3 = T*logdet(R) + tr(R\Syy)
    dl3 = [T*tr(R\dR[:,:,i]) + tr(-(R\dR[:,:,i])*(R\Syy)) for i in 1:length(θ)]
    
    dl1 + dl2 + dl3
end

function em_step(build,θ,z)
    opt = optimize(t->elbo(build(θ),z,t,build),θ)
    Optim.minimizer(opt)
end

function em_step2(build,θ,z;optim_options...)
    m = build(θ)
    kf = kalman_filter(m,z)
    ks = kalman_smoother(kf,m)
    opt = optimize(t->elbo(ks,z,t,build),θ,Optim.Options(;optim_options...))
    Optim.minimizer(opt)
end

struct EMResults
    θs
    ls
end

function em(build,θ0,z;tol=1e-4,max_iterations=1000,verbose=false,optim_options...)
    m0 = build(θ0)
    kf0 = kalman_filter(m0,z)
    ks0 = kalman_smoother(kf0,m0)    
    ls = zeros(max_iterations)
    ls[1] = filter_likelihood(kf0)
    if verbose
            println("Iteration: ",1)
            println("Log likelihood: ",ls[1])
    end
    
    θs = zeros(length(θ0),max_iterations)
    θs[:,1] = θ0[:]
    s = 1
    for i in 2:max_iterations
        θs[:,i] = em_step2(build,θs[:,i-1],z;optim_options...)
        ls[i] = filter_likelihood(kalman_filter(build(θs[:,i]),z))
        if verbose
            println("Iteration: ",i)
            println("Log likelihood: ",ls[i])
        end
        if ls[i] > ls[i-1]
            @warn "Expectation maximization increased the log likelihood"
            break
        end
        s+=1 
        if (ls[i-1] - ls[i])<tol
            break
        end
    end    
    EMResults(θs[:,1:s],ls[1:s])
end
