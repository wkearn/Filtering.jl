# Testing the model-building interface

# All models should have:
# 1. An initial distribution, μ0(θ)
# 2. A transition distribution, f(x,t,θ)
# 3. An observation distribution g(x,t,θ)
# 4. A proposal distribution q(x,y,t,θ)
# 5. A weight function ξ(x,y,t,θ)

δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β = [0.0;1.0]

@testset "LinearStateSpaceModel" begin
    @testset "Generic model interface" begin
        arh = ARH(β)
        @test typeof(arh) == LinearStateSpaceModel

        μ0 = initial_distribution(arh,θ0)

        @test mean(μ0) == [δ/(1-φ)]
        @test cov(μ0) == fill(ση^2/(1-φ^2),1,1)

        x0 = rand(μ0)
        
        f  = transition_distribution(arh,x0,[1.0],1,θ0)

        @test mean(f) == φ*x0 .+ δ
        @test cov(f) == fill(ση^2,1,1)

        x1 = rand(f)
        
        g  = observation_distribution(arh,x1,[1.0],1,θ0)    

        @test mean(g) == x1
        @test cov(g) == fill(σε^2,1,1)

        y1 = rand(g)
    end

    @testset "Linear model interface" begin
        arh = ARH([0.0;1.0])

        μ = initial_mean(arh,θ0)

        @test μ == fill(δ/(1-φ),1)

        Σ0 = initial_covariance(arh,θ0)

        @test Σ0 == fill(ση^2/(1-φ^2),1,1)
        
        F = state_transition_matrix(arh,θ0)

        @test F == fill(φ,1,1)

        Q = state_covariance(arh,θ0)

        @test Q == fill(σ^2,1,1)

        G = state_input_matrix(arh,θ0)

        @test G == fill(δ,1,1)

        H = observation_matrix(arh,θ0)

        @test H == fill(β[2],1,1)

        R = observation_covariance(arh,θ0)

        @test R == fill(σε^2,1,1)

        Γ = observation_input_matrix(arh,θ0)

        @test Γ == fill(β[1],1,1)

        Θ = state_noise_transformation_matrix(arh,θ0)

        @test Θ = fill(1.0,1,1)
    end
end
