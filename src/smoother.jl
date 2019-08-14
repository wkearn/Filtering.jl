export KalmanSmoother, kalman_smoother

struct KalmanSmoother
    x
    P
    Po
    J
    S11
    S00
    S10
    Syy
    z
end

function kalman_smoother(m::LinearStateSpaceModel,z,u=fill([0.0],length(z)))
    kalman_smoother(kalman_filter(m,z,u),m)
end

function kalman_smoother(kf::KalmanFilter,m::LinearStateSpaceModel)
    T = length(kf.xp)
    
    xf = kf.xf
    xp = kf.xp
    Pf = kf.Pf
    Pp = kf.Pp
    K  = kf.K
    
    F = m.F
    H = m.H
    
    x = fill(xf[end],T+1)
    P = fill(Pf[end],T+1)

    # Po[T] = P^T_{T,T-1} = (I-K[T])*F*P^{T-1}_{T-1}
    Po = fill((I-K[T]*H)*F*Pf[T],T)
    
    J = fill(Pf[end],T)
    S = fill(x[end]*x[end]' + P[end],T+1)

    S10 = zeros(size(Pf[end]))
    Syy = zeros(size(H*Pf[end]*H'))
    for t in T:-1:1

        if t == T
            J[t],x[t],P[t],_,S[t],_,syy = smoother_step(x[t+1],P[t+1],xf[t],Pf[t],xp[t],Pp[t],F,Pf[t],J[t],Po[t],H,kf.z[t])
            S10 += x[t+1]*x[t]' + Po[t]
            Syy += syy
        else
            J[t],x[t],P[t],Po[t],S[t],s10,syy = smoother_step(x[t+1],P[t+1],xf[t],Pf[t],xp[t],Pp[t],F,Pf[t+1],J[t+1],Po[t+1],H,kf.z[t])
            S10 += s10
            Syy += syy
        end

        #=
        # J[t] = J_{t-1}
        # x[t] = x^n_{t-1}
        # P[t] = P^n_{t-1}
        # xf[t] = x^{t-1}_{t-1}
        # Pf[t] = P^{t-1}_{t-1}
        # xp[t] = x^{t-1}_t
        # Pp[t] = P^{t-1}_t
        
        # J_{t-1} = P^{t-1}_{t-1}*F'*(P^{t-1}_t)^(-1)
        J[t] = Pf[t]*F'/Pp[t]
        
        # x^n_{t-1} = x^{t-1}_{t-1} + J_{t-1}*(x^n_t - x^{t-1}_t)
        x[t] = xf[t] + J[t]*(x[t+1]-xp[t])
        
        # P^n_{t-1} = P^{t-1}_{t-1} + J_{t-1}*(P^n_t - P^{t-1}_t)*J_{t-1}
        P[t] = Pf[t] + J[t]*(P[t+1]-Pp[t])*J[t]'
        
        # Po[t] = P^n_{t,t-1}
        # = P^{t}_{t} + J_{t}*(P^n_{t+1,t}-F*P^{t}_{t})*J_{t-1}
        if t == T
        else
                Po[t] = Pf[t+1]*J[t]' + J[t+1]*(Po[t+1]-F*Pf[t+1])*J[t]'
        end
        S[t] = x[t]*x[t]' + P[t]
        S10 += x[t+1]*x[t]' + Po[t]
        Syy += (kf.z[t].-H*x[t+1])*(kf.z[t].-H*x[t+1])' .+ H*P[t+1]*H'
        =#
    end

    S11 = sum(S[2:end])
    S00 = sum(S[1:end-1])  
    
    KalmanSmoother(x,P,Po,J,S11,S00,S10,Syy,kf.z)
end

function kalman_smoother(kf::KalmanFilter,m::TimeVariantLinearStateSpaceModel)
    T = length(kf.xp)
    
    xf = kf.xf
    xp = kf.xp
    Pf = kf.Pf
    Pp = kf.Pp
    K  = kf.K
    
    F = m.F
    H = m.H
    
    x = fill(xf[end],T+1)
    P = fill(Pf[end],T+1)

    # Po[T] = P^T_{T,T-1} = (I-K[T])*F*P^{T-1}_{T-1}
    Po = fill((I-K[T]*H[end])*F[end]*Pf[T],T)
    
    J = fill(Pf[end],T)
    S = fill(x[end]*x[end]' + P[end],T+1)

    S10 = zeros(size(Pf[end]))
    Syy = zeros(size(H[end]*Pf[end]*H[end]'))
    for t in T:-1:1

        if t == T
            J[t],x[t],P[t],_,S[t],_,syy = smoother_step(x[t+1],P[t+1],xf[t],Pf[t],xp[t],Pp[t],F[t],Pf[t],J[t],Po[t],H[t],kf.z[t])
            S10 += x[t+1]*x[t]' + Po[t]
            Syy += syy
        else
            J[t],x[t],P[t],Po[t],S[t],s10,syy = smoother_step(x[t+1],P[t+1],xf[t],Pf[t],xp[t],Pp[t],F[t],Pf[t+1],J[t+1],Po[t+1],H[t],kf.z[t])
            S10 += s10
            Syy += syy
        end

    end

    S11 = sum(S[2:end])
    S00 = sum(S[1:end-1])  
    
    KalmanSmoother(x,P,Po,J,S11,S00,S10,Syy,kf.z)
end

function smoother_step(x0,P0,xf,Pf,xp,Pp,F,Pf2,J2,Po2,H,z)
    # J[t] = J_{t-1}
    # x[t] = x^n_{t-1}
    # P[t] = P^n_{t-1}
    # xf[t] = x^{t-1}_{t-1}
    # Pf[t] = P^{t-1}_{t-1}
    # xp[t] = x^{t-1}_t
    # Pp[t] = P^{t-1}_t

    # J_{t-1} = P^{t-1}_{t-1}*F'*(P^{t-1}_t)^(-1)
    J = Pf*F'/Pp
    
    # x^n_{t-1} = x^{t-1}_{t-1} + J_{t-1}*(x^n_t - x^{t-1}_t)
    x = xf + J*(x0-xp)
    
    # P^n_{t-1} = P^{t-1}_{t-1} + J_{t-1}*(P^n_t - P^{t-1}_t)*J_{t-1}
    P = Pf + J*(P0-Pp)*J'
    
    # Po[t] = P^n_{t,t-1}
    # = P^{t}_{t} + J_{t}*(P^n_{t+1,t}-F*P^{t}_{t})*J_{t-1}
    Po = Pf2*J' + J2*(Po2-F*Pf2)*J'
    S = x*x' + P
    S10 = x0*x' + Po
    Syy = (z.-H*x0)*(z.-H*x0)' .+ H*P0*H'
    J,x,P,Po,S,S10,Syy
end
