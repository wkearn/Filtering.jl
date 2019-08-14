export mTFI

function mTFI(θ,m=20)
    σw = θ[1]
    σε = θ[2]
    λ   = exp(θ[3])
    μ0 = θ[4]
    σ0 = θ[5]

    πs = [-ω(j,λ,5/6) for j in 1:m]
    
    F = [πs'; Matrix{Float64}(I,m-1,m-1) zeros(m-1)]
    Θ = [1.0; zeros(m-1,1)]
    G = zeros(1,1)

    H = [1.0 zeros(1,m-1)]
    Γ = zeros(1,1)

    Q = fill(σw^2,1,1)
    R = fill(σε^2,1,1)

    μ = fill(μ0,m)
    Σ0 = σ0^2*Matrix{Float64}(I,m,m)

    LinearStateSpaceModel(F,Q,G,H,R,Γ,Θ,μ,Σ0)
end

# function _gradient(θ,m=20)
#     σw = θ[1]
#     σv = θ[2]
#     σε = θ[3]
#     λ   = exp(θ[4])

#     dQdσw = [2*σw 0; 0 0]
#     dQdσv = [0 0; 0 2*σv]

#     dRdσε = fill(2*σε,1,1)

#     πs = λ*[-dω(j,λ,5/6) for j in 1:m]

#     dFdλ = [zeros(1,m+2); zeros(1,m+2); 0 0 πs'; zeros(m-1,m+2)]

#     dF = cat(zeros(m+2,m+2,3),dFdλ,dims=[3])
#     dQ = cat(dQdσw,dQdσv,zeros(2,2,2),dims=[3])
#     dH = zeros(1,m+2,4)
#     dR = cat(zeros(1,1,2),dRdσε,zeros(1,1),dims=[3])

#     dΘ = zeros(m+2,2,4)

#     dμ = zeros(m+2,1,4)
#     dΣ0 = zeros(m+2,m+2,4)
    
#     LinearStateSpaceGradient(dF,dQ,dH,dR,dΘ,dμ,dΣ0)
# end
