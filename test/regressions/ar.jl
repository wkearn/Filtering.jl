using DelimitedFiles

Random.seed!(1234)

δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β = [0.0;1.0]

N = 1000

@testset "Kalman filter regression tests" begin
    arh = ARH(β)

    X,Y = Filtering.simulate(arh,θ0,N,fill([1.0],N))

    kf = kalman_filter(arh,θ0,Y,fill([1.0],length(Y)))

    @testset "Filter" begin
    
        xf0 = readdlm("regressions/ar/kf_xf.csv")[:,1]
        Pf0 = readdlm("regressions/ar/kf_Pf.csv")[:,1]
        xp0 = readdlm("regressions/ar/kf_xp.csv")[:,1]
        Pp0 = readdlm("regressions/ar/kf_Pp.csv")[:,1]
        Σ0  = readdlm("regressions/ar/kf_Σ.csv")[:,1]
        ε0  = readdlm("regressions/ar/kf_ε.csv")[:,1]

        @test all(vcat(kf.xf...) .== xf0)
        @test all(vcat(kf.Pf...) .== Pf0)
        @test all(vcat(kf.xp...) .== xp0)
        @test all(vcat(kf.Pp...) .== Pp0)
        @test all(vcat(kf.Σ...)  .== Σ0)
        @test all(vcat(kf.ε...)  .== ε0)
    end

    ks = kalman_smoother(arh,θ0,kf)

    @testset "Smoother" begin

        x0  = readdlm("regressions/ar/ks_x.csv")[:,1]
        P0  = readdlm("regressions/ar/ks_P.csv")[:,1]

        @test all(vcat(ks.x...)  .== x0)
        @test all(vcat(ks.P...)  .== P0)
    end

end
