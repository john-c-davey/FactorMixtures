function est_mmvmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
  
    #p = size(Y,2)
    #n = size(Y,1)

    model.w_esteps = zeros(Squantities.n,Squantities.n_esteps,model.g)
    model.Ψ = reshape(repeat(Squantities.psi_initial,model.g),(size(Squantities.psi_initial)[1],model.g))
    Squantities.logL_calc!(model, Mquantities, Squantities)
    if model.status == "failed"
        model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after initialisation"
        model.niter = 0
        return 
    elseif model.status == "failedE"
        model.status = "Failed: E-step for Ψ failed at initialisation" #probably caused by a bad numerical algorithm ... such as finite differences for GH
        model.niter = 0
        return 
    end 
    Mquantities.best_model = deepcopy(model)
    Mquantities.LL_best = copy(model.logL)

    model.niter = Squantities.itmax
    Mquantities.logL_old = model.logL
    model.logL_history[1] = model.logL 
    @inbounds for niter in 1:Squantities.itmax
        model.niter = niter
        Squantities.Mstep_calc!(model, Mquantities, Squantities) #Need to update this function
        if model.status == "failed-MInfNan"
            model.status = "Failed: an Inf or Nan appeared in the M-steps after $niter iterations"
            model.niter = niter
            return 
        end 
        if model.status == "failed"
            model.status = "Failed: got a complex eigenvalue in the M-steps after $niter iterations"
            model.niter = niter
            return 
        end 
        if model.status == "failedΨ"
            model.status = "Failed: M-step update for Ψ failed after $niter iterations (probably due to paused E-steps)"
            model.niter = niter
            return 
        end
        if (Mquantities.ecm_failed) & (model.name == "MGHFA")
            @warn "AECM updating step used on iteration $niter"
            push!(model.update_history,"AECM")
        else
            push!(model.update_history,"ECM")
        end
        Squantities.logL_calc!(model, Mquantities, Squantities)
        if model.status == "failed"
            model.status = "Failed: calculation of log-likelihod failed after $niter iterations"
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Failed; calculation of log-likelihod failed after $niter iterations; Returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        elseif model.status == "failedE"
            model.status = "Failed: E-step for Ψ failed after $niter iterations" #probably caused by a bad numerical algorithm ... such as finite differences for GH
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Failed; E-step for Ψ failed after $niter iterations; Returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break 
        end 

        if model.logL > Mquantities.LL_best
            Mquantities.LL_best = copy(model.logL)
            Mquantities.best_model = deepcopy(model)
        end


        if ((Squantities.conv_measure == "diff") && (abs(model.logL - Mquantities.logL_old) < Squantities.tol))
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        end 

        if ((Squantities.conv_measure == "ratio") && (abs((model.logL - Mquantities.logL_old)/model.logL) < Squantities.tol))
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        end
        #print(niter)
        #if niter == 10
        #    print(model)
        #end
        Mquantities.logL_old = model.logL
        model.logL_history[niter + 1] = model.logL 
        if model.logL < Mquantities.LL_best 
            model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
            model.B = Mquantities.best_model.B
            model.D = Mquantities.best_model.D
            model.mu = Mquantities.best_model.mu
            model.pivec = Mquantities.best_model.pivec
            model.logL = Mquantities.best_model.logL
        end
    end #for 
    #model.niter = Squantities.itmax
end 

function est_msmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
  
    model.w_esteps = zeros(Squantities.n,Squantities.n_esteps,model.g)
    model.Ψ = reshape(repeat(Squantities.psi_initial,model.g),(size(Squantities.psi_initial)[1],model.g))
    model.beta = zeros(Squantities.p,model.g)
    model.niter = Squantities.itmax
    Squantities.logL_calc!(model, Mquantities, Squantities)
    if model.status == "failed"
        model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after initialisation"
        return 
    end 
    Mquantities.logL_old = model.logL
    model.logL_history[1] = model.logL 
    @inbounds for niter in 1:Squantities.itmax
        Squantities.Mstep_calc!(model, Mquantities, Squantities)
        if model.status == "failed-MInfNan"
            model.status = "Failed: an Inf or Nan appeared in the M-steps after $niter iterations"
            model.niter = niter
            return 
        end 
        if model.status == "failed"
            model.status = "Failed: got a complex eigenvalue in the M-steps after $niter iterations"
            model.niter = niter
            return 
        end 
        if model.status == "failedΨ"
            model.status = "Failed: M-step update for Ψ failed after $niter iterations (probably due to paused E-steps)"
            model.niter = niter
            return 
        end
        Squantities.logL_calc!(model, Mquantities, Squantities)
        if model.status == "failed"
            model.status = "Failed: calculation of log-likelihod failed after $niter iterations"
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Failed; calculation of log-likelihod failed after $niter iterations; Returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        elseif model.status == "failedE"
            model.status = "Failed: E-step for Ψ failed after $niter iterations" #probably caused by a bad numerical algorithm ... such as finite differences for GH
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Failed; E-step for Ψ failed after $niter iterations; Returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break 
        end 

        if model.logL > Mquantities.LL_best
            Mquantities.LL_best = copy(model.logL)
            Mquantities.best_model = deepcopy(model)
        end

        if ((Squantities.conv_measure == "diff") && (abs(model.logL - Mquantities.logL_old) < Squantities.tol))
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        end 

        if ((Squantities.conv_measure == "ratio") && (abs((model.logL - Mquantities.logL_old)/model.logL) < Squantities.tol))
            model.niter = niter
            if model.logL < Mquantities.LL_best 
                model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
                model.B = Mquantities.best_model.B
                model.D = Mquantities.best_model.D
                model.mu = Mquantities.best_model.mu
                model.pivec = Mquantities.best_model.pivec
                model.logL = Mquantities.best_model.logL
            end
            break
        end
        #print(niter)
        #if niter == 10
        #    print(model)
        #end
        Mquantities.logL_old = model.logL
        model.logL_history[niter + 1] = model.logL 
        if model.logL < Mquantities.LL_best 
            model.status = "Completed; returning model with highest log-likelihood not model obtained at termination condition."
            model.B = Mquantities.best_model.B
            model.D = Mquantities.best_model.D
            model.mu = Mquantities.best_model.mu
            model.pivec = Mquantities.best_model.pivec
            model.logL = Mquantities.best_model.logL
        end
    end #for 
    #model.niter = Squantities.itmax
    #if model == "failedModel"
    #    return "failedModel"
    #else 
    #    return model
    #end
end 

function est_mfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
  
    #p = size(Y,2)
    #n = size(Y,1)
    model.beta =  zeros(Squantities.p,model.g)
    #init_para.beta = zeros(p,init_para.g)
    model.niter = Squantities.itmax
    Squantities.logL_calc!(model, Mquantities, Squantities) 
    if model.status == "failed"
        model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after initialisation"
        return 
    end 
    Mquantities.logL_old = model.logL
    model.logL_history[1] = model.logL 
    @inbounds for niter in 1:Squantities.itmax
        Squantities.Mstep_calc!(model, Mquantities, Squantities) #Need to update this function
        if model.status == "failed-MInfNan"
            model.status = "Failed: an Inf or Nan appeared in the M-steps after $niter iterations"
            model.niter = niter
            return 
        end 
        if model.status == "failed"
            model.status = "Failed: got a complex eigenvalue in the M-steps after $niter iterations"
            model.niter = Squantities.itmax
            return 
        end 
  
        Squantities.logL_calc!(model, Mquantities, Squantities)
        if model.status == "failed"
            model.status = "Failed: got a NaN in a log-likelihod or E-step for Ψ after $niter iterations"
            model.niter = Squantities.itmax
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
    end #for 
end 