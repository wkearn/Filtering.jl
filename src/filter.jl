export KalmanFilter, kalman_filter, filter_likelihood

struct KalmanFilter
    xf
    xp
    Pf
    Pp
    K
    ε
    Σ
    z
end

"""
Apply a Kalman filter specified 
by a LinearStateSpaceModel to a time series, `z`.
"""
function kalman_filter(m::LinearStateSpaceModel,z,u=fill([0.0],length(z)))
    T = length(z)

    μ = m.μ
    Σ0 = m.Σ0
    F = m.F
    G = m.G
    Q = m.Q
    
    Θ = m.Θ
    Qp = Θ*Q*Θ'

    H = m.H
    R = m.R
    Γ = m.Γ

    
    d = length(μ)

    dt = eltype(F*μ)
    xf = fill(dt.(μ),T+1)
    xp = fill(zeros(dt,d),T)
    Pf = fill(dt.(Σ0),T+1)
    Pp = fill(zeros(dt,d,d),T)
    Σ = fill(zeros(dt,d,d),T)
    K = fill(zeros(dt,d,d),T)
    ε = fill(zeros(dt,d),T)
    for t in 1:T
        x1,P1,ε1,Σ1,K1,xf1,Pf1 = filter_step(z[t],u[t],F,G,Qp,H,Γ,R,xf[t],Pf[t])
        xp[t] = x1
        Pp[t] = P1
        ε[t]  = ε1
        Σ[t]  = Σ1
        K[t]  = K1
        xf[t+1]= xf1
        Pf[t+1] = Pf1
    end
    KalmanFilter(xf,xp,Pf,Pp,K,ε,Σ,z)
end

"""
Apply a Kalman filter specified 
by a LinearStateSpaceModel to a time series, `z`.
"""
function kalman_filter(m::TimeVariantLinearStateSpaceModel,z,u=fill([0.0],length(z)))
    T = length(z)

    μ = m.μ
    Σ0 = m.Σ0
    F = m.F
    G = m.G
    Q = m.Q
    
    Θ = m.Θ
    Qp = [Θ[t]*Q[t]*Θ[t]' for t in eachindex(Θ)]

    H = m.H
    R = m.R
    Γ = m.Γ

    
    d = length(μ)
    
    xf = fill(μ,T+1)
    xp = fill(zeros(d),T)
    Pf = fill(Σ0,T+1)
    Pp = fill(zeros(d,d),T)
    K = fill(zeros(d,d),T)
    Σ = fill(zeros(d,d),T)
    ε = fill(zeros(d),T)
    
    for t in 1:T
        xp[t],Pp[t],ε[t],Σ[t],K[t],xf[t+1],Pf[t+1] = filter_step(z[t],u[t],F[t],G[t],Qp[t],H[t],Γ[t],R[t],xf[t],Pf[t])
    end
    KalmanFilter(xf,xp,Pf,Pp,K,ε,Σ,z)
end

function filter_step(z,u,F,G,Qp,H,Γ,R,xf0,Pf0)
    xp = F*xf0 + G*u
    Pp = F*Pf0*F' .+ Qp
        
    ε = z .- H*xp .- Γ*u
    Σ = H*Pp*H'.+R    
    K = Pp*H'*inv(Σ)
        
    xf = xp .+ K * ε
    Pf = (I-K*H)*Pp

    xp,Pp,ε,Σ,K,xf,Pf
end

"""
Calculate the negative log-likelihood 
using the innovations representation
"""
function filter_likelihood(kf::KalmanFilter)
    0.5*sum(logdet.(kf.Σ) + [kf.ε[i]'*(kf.Σ[i]\kf.ε[i]) for i in eachindex(kf.ε)] .+ length(kf.ε[1])*log(2pi))
end
