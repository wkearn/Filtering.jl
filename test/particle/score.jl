# Tests for score filter

δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β = [0.0;1.0]

N = 1000

@testset "Score filter" begin
    arh = ARH(β)

    X,Y = Filtering.simulate(arh,θ0,N,fill([1.0],N))

    sf = score_filter(arh,θ0,Y,100,fill([1.0],N))

    @test loglikelihood(sf) < 0.0
    @test size(score(sf),1) == length(θ0)    
end
