# Tests for score filter

using ForwardDiff

δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β = [0.0;1.0]

N = 100

@testset "Score filter" begin
    arh = ARH(β)

    X,Y = Filtering.simulate(arh,θ0,N,fill([1.0],N))

    S = ForwardDiff.gradient(θ->-filter_likelihood(kalman_filter(arh,θ,Y,fill([1.0],N))),θ0)

    sf1 = score_filter(arh,θ0,Y,100,fill([1.0],N))
    sf2 = score_filter(arh,θ0,Y,1000,fill([1.0],N))
    sf3 = score_filter(arh,θ0,Y,10000,fill([1.0],N))

    @test loglikelihood(sf1) < 0.0
    @test size(score(sf1),1) == length(θ0)

    # This might not always succeed, but should be true
    @test mean(abs2,S .- score(sf1)) > mean(abs2,S .- score(sf2)) > mean(abs2,S .- score(sf3))
end
