function logL_mfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)

    #Fji = @MMatrix zeros(Float64,Squantities.n,model.g)

    @inbounds for i in 1:model.g 
        
        #Mquantities.sigma = real(model.B[:,:,i]*model.B[:,:,i]') + model.D[:,:,i]
        #Mquantities.Fji[:,i] = logpdf(MvNormal(Vector(model.mu[:,i]), Matrix(Mquantities.sigma)), Squantities.Y')
        Mquantities.sigma = Symmetric(real(model.B[:,:,i]*model.B[:,:,i]') + model.D[:,:,i])
        Mquantities.delta = try
            Distances.colwise(SqMahalanobis(Symmetric(inv(Mquantities.sigma))), Array(Squantities.Y'), model.mu[:,i])
        catch e 
            model.status = "failed"
            return 
        end
        if det(Mquantities.sigma) < 0
            model.status = "failed"
            return 
        end


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


    #Mquantities.Fji = Mquantities.Fji .+ log.(model.pivec)
    #Mquantities.Fjmax = maximum(Mquantities.Fji, dims = 2)
    #Mquantities.Fji = Mquantities.Fji .- Mquantities.Fjmax 
    #model.logL = sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
    #Mquantities.Fji = exp.(Mquantities.Fji)
    #model.tau = Mquantities.Fji ./ sum(Mquantities.Fji, dims = 2)

    Fji_sum = copy(Mquantities.Fji) .+ log.(model.pivec)
    #Mquantities.Fjmax = maximum(Fji_sum, dims = 2)
    #Fji_subtract = copy(Fji_sum) .- Mquantities.Fjmax
    model.logL = sum(log.(sum(exp.(Fji_sum), dims = 2))) #sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
    #Fji_exp = exp.(Fji_subtract)
    model.tau = exp.(Fji_sum) ./ sum(exp.(Fji_sum), dims = 2)
end

function logL_mmvmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
    #n,p = size(Y)
    #p = size(Y,2)
    #n = size(Y,1)
    #Fji = @MMatrix zeros(Float64,n,model.g)

    

    @inbounds for i in 1:model.g 
        oldFji = copy(Mquantities.Fji[:,i])
        oldw_esteps = copy(model.w_esteps[:,:,i])
        try
            Squantities.logfY!(model, Mquantities, Squantities, i)
        catch
            model.status = "failed"
            return
        end
        try
            Squantities.esteps!(model, Mquantities, Squantities, i)
        catch
            model.status = "failedE"
            return
        end
        if (any(isinf.(Mquantities.Fji[:, i])))
            model.status = "failed"
            return
        end

        if (any(isnan.(Mquantities.Fji[:, i])) | any(isnan.(model.w_esteps[:,:,i])))
            #Mquantities.na_log_esteps = Vector{Int64}()
            #model.status = "failed"
            baddp_mean = [x[1] for x in findall(x->x<=0.1, Distances.pairwise(Euclidean(), Squantities.Y', model.mu))]
            #badmean = [x[2] for x in findall(x->x<=0.1, Distances.pairwise(Euclidean(), Squantities.Y', model.mu))]
            baddp_weight = [x[1] for x in findall(x->x>0, isnan.(model.w_esteps[:,:,i]))]
            Mquantities.na_log_esteps = unique(vcat(baddp_mean,baddp_weight))

            for bad in Mquantities.na_log_esteps
                if !(bad in Mquantities.estep_rolled_points)
                    push!(Mquantities.estep_rolled_points, bad)
                    if length(baddp_weight)  > 0
                        @warn "There was a problem estimating an E-step for data point $bad, rolled over E-steps from previous values for that point."
                    elseif length(baddp_mean)  > 0
                        @warn "A component mean appears to have converged to data point $bad, rolled over E-steps from previous values for that point."
                    end 
                end
            end

            #Squantities.Y[setdiff(1:end, unique_bad_dp ), :]
            
            #return 
            Mquantities.Fji[Mquantities.na_log_esteps,i] = copy(oldFji[Mquantities.na_log_esteps])
            model.w_esteps[Mquantities.na_log_esteps,:,i] = copy(oldw_esteps[Mquantities.na_log_esteps,:])

        end

    end

    #if length(Mquantities.na_log_esteps) > 0
    #    model.logL = NaN
    #    Mquantities.Fji = Mquantities.Fji .+ log.(model.pivec)
    #    Mquantities.Fjmax = maximum(Mquantities.Fji, dims = 2)
    #    Mquantities.Fji = Mquantities.Fji .- Mquantities.Fjmax 
    #    Mquantities.Fji = exp.(Mquantities.Fji)
    ##    model.tau = Mquantities.Fji ./ sum(Mquantities.Fji, dims = 2)
     ##   model.tau[Mquantities.na_log_esteps,:] .= 0 
      #  model.w_esteps[Mquantities.na_log_esteps,:,:] .= 0  
      #  Mquantities.na_log_esteps = Vector{Int64}()
    #else

        
    #This does the correct calculation ...  
        #Mquantities.Fji = copy(Mquantities.Fji) .+ log.(model.pivec)
        Fji_sum = copy(Mquantities.Fji) .+ log.(model.pivec)
        #Mquantities.Fjmax = maximum(Fji_sum, dims = 2)
        #Fji_subtract = copy(Fji_sum) .- Mquantities.Fjmax
        model.logL = sum(log.(sum(exp.(Fji_sum), dims = 2))) #sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
        #Fji_exp = exp.(Fji_subtract)
        model.tau = exp.(Fji_sum) ./ sum(exp.(Fji_sum), dims = 2)
    #end

    # log likelihood calc was pretty off, checking if code from MFA fixes it...!
    # Mquantities.Fji = Mquantities.Fji .+ log.(model.pivec)
    #Mquantities.Fjmax = maximum(Mquantities.Fji, dims = 2)
    #Mquantities.Fji = Mquantities.Fji .- Mquantities.Fjmax 
    #model.logL = sum(Mquantities.Fjmax) + sum(log.(sum(exp.(Mquantities.Fji), dims = 2)))
    #Mquantities.Fji = exp.(Mquantities.Fji)
    #model.tau = Mquantities.Fji ./ sum(Mquantities.Fji, dims = 2)


end