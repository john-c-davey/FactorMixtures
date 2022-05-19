function est_mfa_aecm(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
  
    #p = size(Y,2)
    #n = size(Y,1)
    model.beta =  zeros(Squantities.p,model.g)
    #init_para.beta = zeros(p,init_para.g)
    model.logL_history = Vector{Float64}(undef, 2*Squantities.itmax+1)
    Squantities.logL_calc!(model, Mquantities, Squantities) 
    if model.status == "failed"
        model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after initialisation"
        return 
    end 
    Mquantities.logL_old = model.logL
    model.logL_history[1] = model.logL 
    @inbounds for niter in 1:Squantities.itmax
        Squantities.Mstep_calc!(model, Mquantities, Squantities, 1) #Need to update this function
        if model.status == "failed"
            model.status = "Failed: got a complex eigenvalue in the M-steps after $niter iterations"
            return 
        end 
  
        Squantities.logL_calc!(model, Mquantities, Squantities)
        if model.status == "failed"
            model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after $niter iterations"
            return 
        end 
        
        Squantities.Mstep_calc!(model, Mquantities, Squantities, 2) #Need to update this function
        if model.status == "failed"
            model.status = "Failed: got a complex eigenvalue in the M-steps after $niter iterations"
            return 
        end 
  
        Squantities.logL_calc!(model, Mquantities, Squantities)
        if model.status == "failed"
            model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after $niter iterations"
            return 
        end 


        if ((Squantities.conv_measure == "diff") && (abs(model.logL - Mquantities.logL_old) < Squantities.tol))
            model.niter = niter
            model.logL_history[niter + 1] = model.logL 
            break
        end 
  
        if ((Squantities.conv_measure == "ratio") && (abs((model.logL - Mquantities.logL_old)/model.logL) < Squantities.tol))
            model.niter = niter
            model.logL_history[niter + 1] = model.logL 
            break
        end
  
        Mquantities.logL_old = model.logL
        model.logL_history[niter + 1] = model.logL 
        if niter == Squantities.itmax
            model.niter = niter
        end
    end #for 
    
end 

function Mstep_calc_aecm(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities, step::Int64)
    if step==1
        Mquantities.n_i = sum(model.tau, dims = 1)
        #AECM-Step 1: Updating pi and mu_i
        model.pivec = Mquantities.n_i ./ Squantities.n 
        @inbounds for i in 1:model.g
            @views model.mu[:,i] = sum(Squantities.Y .* model.tau[:,i], dims = 1) ./ sum(model.tau[:,i]) 
        end 
    else
        @inbounds for i in 1:model.g
        Ymu = Squantities.Y .- model.mu[:,i]' 
        V = (Array(Ymu') * (Ymu .* model.tau[:,i]))/Mquantities.n_i[i]
        gamma = inv(real.(model.B[:,:,i]*model.B[:,:,i]' + model.D[:,:,i]))*model.B[:,:,i]
        omega = I(model.q) - gamma' * model.B[:,:,i]
        model.B[:,:,i] =  V*gamma*inv(gamma'*V*gamma + omega)
        model.D[:,:,i] = diagm(diag(V - V*gamma*model.B[:,:,i]'))
        end
    end
end
