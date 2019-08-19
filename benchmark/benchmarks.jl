using BenchmarkTools, Filtering


SUITE = BenchmarkGroup()

SUITE["ar_linear"] = BenchmarkGroup(["models","linear",])

β = [0.0;1.0]
δ = 1.0
φ = 0.95
ση = 1.0
σε = 1.0

θ0 = [δ;atanh(φ);ση;σε]

β = [0.0;1.0]

N = 1000

arh = ARH(β)

SUITE["ar_linear"]["simulate"] = @benchmarkable Filtering.simulate(arh,θ0,1000)

X,Y = Filtering.simulate(arh,θ0,1000)

SUITE["ar_linear"]["kalman"] = @benchmarkable kalman_filter(arh,θ0,Y,fill([1.0],length(Y)))
SUITE["ar_linear"]["particle"] = @benchmarkable particle_filter(arh,θ0,Y,N,fill([1.0],length(Y)))


