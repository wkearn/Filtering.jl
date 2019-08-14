export AR1, AR1TV, ARH

function AR1(θ)
    μ = fill(θ[1],1)
    Σ0 = fill(θ[2].^2,1,1)
    F = fill(θ[3],1,1)
    G = zeros(1,1)
    Q = fill(θ[4].^2,1,1)
    H = [1.0]'
    R = fill(θ[5].^2,1,1)
    Γ = zeros(1,1)
    Θ = fill(1.0,1,1)
    LinearStateSpaceModel(F,Q,G,H,R,Γ,Θ,μ,Σ0)
end

function AR1TV(θ,N)
    μ = fill(θ[1],1)
    Σ0 = fill(θ[2].^2,1,1)
    F = fill(fill(θ[3],1,1),N)
    G = fill(zeros(1,1),N)
    Q = fill(fill(θ[4].^2,1,1),N)
    H = fill([1.0]',N)
    R = fill(fill(θ[5].^2,1,1),N)
    Γ = fill(zeros(1,1),N)
    Θ = fill(fill(1.0,1,1),N)
    TimeVariantLinearStateSpaceModel(F,Q,G,H,R,Γ,Θ,μ,Σ0)
end

"""
An autoregressive model with specified regression coefficients

    - θ = [δ;atanh(φ);σ]
    - β = [α1;α2;σε]

x_{t} = δ + φ*x_{t-1} + σ η_t
y_{t} = α1+ α2*x_t + σε ε_t


"""
function ARH(θ,β)
    δ = θ[1]
    φ = tanh(θ[2])
    σ = θ[3]
    σε = θ[4]
    α1= β[1]
    α2= β[2]

    μ = fill(δ/(1-φ),1)
    Σ0= fill(σ^2/(1-φ^2),1,1)
    F = fill(φ,1,1)
    Q = fill(σ^2,1,1)
    G = fill(δ,1,1)
    H = fill(α2,1,1)
    R = fill(σε^2,1,1)
    Γ = fill(α1,1,1)
    Θ = fill(1.0,1,1)    
    
    LinearStateSpaceModel(F, Q, G, H, R, Γ, Θ, μ, Σ0)
end
