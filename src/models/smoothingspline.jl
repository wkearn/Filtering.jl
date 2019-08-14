export SmoothingSpline, SmoothingSpline2

function SmoothingSpline(θ)
    F = [2.0 -1.0; 1.0 0.0]
    Q = fill(θ[1].^2,1,1)
    G = zeros(2,1)
    H = [1.0 0.0]
    R = fill(θ[2].^2,1,1)
    Γ = zeros(1,1)
    Θ = [1.0;0.0]
    μ = [0.0;0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    LinearStateSpaceModel(F,Q,G,H,R,Γ,Θ,μ,Σ0)
end
