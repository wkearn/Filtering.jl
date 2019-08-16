# Particle filtering an AR(1) model with measurement noise

μ0(θ) = MvNormal([θ[1]/(1-tanh(θ[2]))],fill(θ[3]^2/(1-tanh(θ[2])^2),1,1))
f(x,t,θ) = MvNormal(θ[1] .+ tanh(θ[2])*x,fill(θ[3]^2,1,1))
g(x,t,θ) = MvNormal(x,fill(θ[4]^2,1,1))

function q(x,y,t,θ)
    δ = θ[1]
    φ = tanh(θ[2])
    σ = θ[3]
    σε = θ[4]
    ν = δ .+ φ * x
    MvNormal((ν*σε^2 .+ y*σ^2)/(σε^2 + σ^2),fill((σ^2*σε^2)/(σ^2 + σε^2),1,1))
end

function ξ(x,y,t,θ)
    δ = θ[1]
    φ = tanh(θ[2])
    σ = θ[3]
    σε = θ[4]
    ν = δ .+ φ * x
    logpdf(MvNormal(ν,sqrt(σ^2 + σε^2)),y)    
end

mar = ProposalStateSpaceModel(μ0,f,g,q,ξ)

δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β  = [0.0;1.0]

T = 1000
N = 100

Random.seed!(1234)

X,Y = Filtering.simulate(mar,θ0,T)
X1 = X[1][1]

Random.seed!(1234)

@testset "Simulating from model" begin
    X,Y = Filtering.simulate(mar,θ0,T)

    # Test to ensure that the RNG seed is working
    @test X[1][1] == X1

    @test length(X) == T+1
    @test length(Y) == T

    μ = δ/(1-φ)
    σ = ση/sqrt(1-φ^2)

    # These are rough approximations, just to make sure we are
    # getting a time series that looks right. Perhaps they should
    # be replaced with proper hypothesis tests.
    
    @test abs(mean(hcat(X...)[1,:]) - μ) < 3.0 
    @test abs(std(hcat(X...)[1,:]) - σ) < 2.0
    @test abs(autocor(hcat(X...)[1,:],[1])[1] - φ) < 0.2
end

@testset "Running Kalman filter" begin
    X,Y = Filtering.simulate(mar,θ0,T)

    @test X[1][1] == X1
    
    arh = ARH(β)
    kf = kalman_filter(arh,θ0,Y,map(x->[x],ones(T)))
    ks = kalman_smoother(arh,θ0,kf)
end

@testset "Running particle filter" begin
    X,Y = Filtering.simulate(mar,θ0,T)

    @test X[1][1] == X1
    
    p1 = Filtering.particle_filter(mar,Y,θ0,N)
    #p2 = Filtering.filtering_filter(mar,Y,θ0,N)
end

#=
@testset "Sampling from smoothing distribution" begin
    B = 1

    Xs = [Filtering.generate_realization(p2,mar,θ0) for i in 1:B]
end
=#  
