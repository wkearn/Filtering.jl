This package collects code related to filtering and smoothing in state-space models. The goal is to provide a single interface for efficiently filtering many different models. A secondary goal is to provide facilities for parameter inference in the state-space models, focusing particularly on likelihood and quasi-likelihood methods. 

One can think of this package as a restricted probabilistic programming language, specialized to working with state-space models. Other PPLs, such as [Turing.jl](https://github.com/TuringLang/Turing.jl), can handle more complex probabilistic models.

For linear, Gaussian state-space models, the `LinearStateSpaceModel` type specifies the model, which can be used by `kalman_filter` and `kalman_smoother` to filter and smooth a time series. 

For nonlinear or non-Gaussian models, the `ProposalStateSpaceModel` specifies a model in terms of `Distributions` from Distributions.jl. The `ProposalStateSpaceModel` also requires a proposal distribution and a weighting function, which enables its use with an auxiliary particle filter. For convenience a `BootstrapStateSpaceModel` is provide, which implements the bootstrap particle filter as a special case of the auxiliary particle filter and which does not need a specified proposal distribution or weighting function. A forward-backward particle smoother is also implemented (but is currently quite slow). Variants of the particle filter that do not calculate the score function or that store the filtering distribution at each time step are also provided. 

Parameter estimation for a `LinearStateSpaceModel` can be achieved by optimizing the filter log-likelihood with Optim.jl. Parameter estimation for nonlinear, non-Gaussian models is currently implemented with the particle approximation of the score function given by Nemeth et al. (2017). The estimated score function can be applied in a stochastic gradient descent algorithm to optimize the log-likelihood, and several SGD variants are implemented.





