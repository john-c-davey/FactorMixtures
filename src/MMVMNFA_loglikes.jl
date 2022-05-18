function logL_mfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)

    #Fji = @MMatrix zeros(Float64,Squantities.n,model.g)

    @inbounds for i in 1:model.g 
        
        #Mquantities.sigma = real(model.B[:,:,i]*model.B[:,:,i]') + model.D[:,:,i]
        #Mquantities.Fji[:,i] = logpdf(MvNormal(Vector(model.mu[:,i]), Matrix(Mquantities.sigma)), Squantities.Y')
        Mquantities.sigma = real(model.B[:,:,i]*model.B[:,:,i]') + model.D[:,:,i]
        Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,i])


        #@views inv_D = Diagonal(1 ./ model.D[:,:,i])
        #@views B_inv_D = model.B[:,:,i] .* diag(inv_D)
        #@views inv_O = inv( I(model.q) + B_inv_D' * model.B[:,:,i] )
        #inv_S = inv_D - B_inv_D * inv_O * B_inv_D'
        #@views logdetS = sum(log.(diag(model.D[:,:,i]))) - log(det(inv_O))

        #@views mahal_dist = Distances.colwise(SqMahalanobis(inv_S), Array(Y'), model.mu[:,i])

        #Fji[:, i] = -0.5 .* mahal_dist .- (p/2) .* log(2 * pi) .- 0.5 .* logdetS
        Mquantities.Fji[:, i] = -0.5 * Mquantities.delta .- 0.5*Squantities.p*log(2*pi) .- 0.5*log(det(Mquantities.sigma))
        if(any(isnan.(Mquantities.Fji[:, i])))
            model.status = "failed"
            return 
        end

    end

    Mquantities.Fji = Mquantities.Fji .+ log.(model.pivec)
    Mquantities.Fjmax = maximum(Mquantities.Fji, dims = 2)
    Mquantities.Fji = Mquantities.Fji .- Mquantities.Fjmax 
    model.logL = sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
    Mquantities.Fji = exp.(Mquantities.Fji)
    model.tau = Mquantities.Fji ./ sum(Mquantities.Fji, dims = 2)

end

function logL_mmvmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
    #n,p = size(Y)
    #p = size(Y,2)
    #n = size(Y,1)
    #Fji = @MMatrix zeros(Float64,n,model.g)
    @inbounds for i in 1:model.g 
        
        Squantities.logfY!(model, Mquantities, Squantities, i)
        Squantities.esteps!(model, Mquantities, Squantities, i)
        if(any(isnan.(Mquantities.Fji[:, i])) | any(isnan.(model.w_esteps[:,:,i])))
            model.status = "failed"
            return 
        end

    end

    Mquantities.Fji = Mquantities.Fji .+ log.(model.pivec)
    Mquantities.Fjmax = maximum(Mquantities.Fji, dims = 2)
    Mquantities.Fji = Mquantities.Fji .- Mquantities.Fjmax 
    model.logL = sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
    Mquantities.Fji = exp.(Mquantities.Fji)
    model.tau = Mquantities.Fji ./ sum(Mquantities.Fji, dims = 2)

end