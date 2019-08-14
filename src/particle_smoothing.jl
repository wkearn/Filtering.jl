function generate_realization(pc::FilteringParticleContainer,m::ProposalStateSpaceModel,θ,sample_function=resample_stratified)

    D = size(pc.X,1)
    N = size(pc.X,2)
    T = size(pc.X,3)-1
    
    x = zeros(D,T+1)
    x[:,end] = pc.X[:,sample_function(exp.(pc.w[:,end])./sum(exp,pc.w[:,end]),1),end]
   
    for t in T-1:-1:0
        wsmooth = pc.w[:,t+1]
        for i in 1:N
            # This is quite slow
            wsmooth[i] += logpdf(m.f(pc.X[:,i,t+1],t,θ),x[:,t+2])
        end
        
        x[:,t+1] = pc.X[:,sample_function(exp.(wsmooth)./sum(exp,wsmooth),1),t+1]
    end
    x
end
