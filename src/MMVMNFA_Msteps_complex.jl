function MStep_mfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
    #update = MFAModelM(model.pivec,model.mu,model.B,model.D,model.g,model.q,model.tau,model.logL,model.BIC)
    Mquantities.n_i = sum(model.tau, dims = 1)
     
    #CM-Step 1: Updating pi and mu_i
    model.pivec = Mquantities.n_i ./ Squantities.n 
    #mu_new = Array{Float64,2}(undef,p,g)
    @inbounds for i in 1:model.g
        @views model.mu[:,i] = sum(Squantities.Y .* model.tau[:,i], dims = 1) ./ sum(model.tau[:,i]) 
    end 

    #CM-Step 2: Updating q 
    #Vtilde = @MArray zeros(Float64,p,p,model.g)
    #Lambda = @MMatrix zeros(Float64,q_max,model.g)
    #Lambda_new = Array{Float64,2}(undef,q_max,model.g)
    #eigenvecs =  Array{Any,1}(undef,model.g)
    @inbounds for i in 1:model.g
        @views Mquantities.Ymu = Squantities.Y .- model.mu[:,i]' 
        @views Mquantities.Vi = (Array(Mquantities.Ymu') * (Mquantities.Ymu .* model.tau[:,i]))/Mquantities.n_i[i]
        @views Mquantities.D_is = Diagonal(1 ./ sqrt.(model.D[:,:,i]))
        Mquantities.Vtilde[:,:,i] = Mquantities.D_is*Mquantities.Vi*Mquantities.D_is
        #@views Mquantities.V_eig_temp = eigvals(Array(Mquantities.Vtilde[:,:,i]))
        #if any(imag.(V_eig_temp) .!= 0)   
        #    print("uh oh")
        #    print(imag.(V_eig_temp))
        #    print("converting to real number >:(")
        #end 
        @views Mquantities.V_eig_temp = try
            eigvals(Array(Mquantities.Vtilde[:,:,i]))
        catch e
            model.status = "failed-MInfNan"
            return 
        end

        if any(imag(Mquantities.V_eig_temp) .> 0)
            println("Complex eigenvalues appeared!!")
            model.status = "failed"
            return
        end 

       # Mquantities.V_eig = real.(Mquantities.V_eig_temp)
        Mquantities.eig_order = sortperm(Mquantities.V_eig_temp, rev = true)
        @views Mquantities.Lambda[:,i] = Mquantities.V_eig_temp[Mquantities.eig_order][1:Squantities.ledermann]
        @views Mquantities.eigenvecs[i] = real.(eigvecs(Array(Mquantities.Vtilde[:,:,i]))[:,Mquantities.eig_order])
        if any(Mquantities.Lambda[:,i] .< 0)
            println("negative eigenvalues appeared!!")
            Mquantities.Lambda[(Mquantities.Lambda[:,i] .< 0),i] .= 1e-20
        end
    end
    #CM-Step 2.5: Optionally updating q
    if Squantities.update_q #Fix this so that we don't accidentally change the value of q when we shouldnt.
        Mquantities.qup.Lambda_cs = cumsum(log.(Mquantities.Lambda) - Mquantities.Lambda + ones(size(Mquantities.Lambda)), dims = 1)
        Mquantities.qup.n_param = Int.((model.g - 1) .+ 2*model.g*Squantities.p .+ model.g * (Squantities.p .* (1:Squantities.ledermann) .- (1:Squantities.ledermann) .* ((1:Squantities.ledermann) .- 1)/2))
        Mquantities.qup.q_scores = sum(Mquantities.qup.Lambda_cs .* Mquantities.n_i, dims = 2) .+ log(Squantities.n)*Mquantities.qup.n_param
        Mquantities.qup.q_up = argmin(vec(Mquantities.qup.q_scores))
        Mquantities.qup.q_change = (Mquantities.qup.q_up != model.q)
        model.q = Mquantities.qup.q_up
        if Mquantities.qup.q_change
            Mquantities.qup.B_update = zeros(ComplexF64,Squantities.p,model.q,model.g)
        end
    end

    #CM-Step 3: Updating B_i, then D_i in same loop.
    Mquantities.qi_break = false 
    @inbounds for i in 1:model.g
        #Update B
        @views Mquantities.qi = sum(Mquantities.Lambda[:,i] .> 1)
        if Mquantities.qi > model.q 
            Mquantities.qi = model.q
        end 

        if Mquantities.qi == 0
            Mquantities.qi_break = true 
           break
        end 
        @views Mquantities.Uqi = Mquantities.eigenvecs[i][:,1:Mquantities.qi]
        @views Mquantities.Ri = I(model.q)[1:Mquantities.qi,:]
        if Squantities.update_q 
            if Mquantities.qup.q_change
                @views Mquantities.qup.B_update[:,:,i] = sqrt.(model.D[:,:,i]) * Mquantities.Uqi * 
                Diagonal(sqrt.(Mquantities.Lambda[1:Mquantities.qi,i] - vec(ones(1,Mquantities.qi)))) * Mquantities.Ri
            else
                @views model.B[:,:,i] = sqrt.(model.D[:,:,i]) * Mquantities.Uqi * 
            Diagonal(sqrt.(Mquantities.Lambda[1:Mquantities.qi,i] - vec(ones(1,Mquantities.qi)))) * Mquantities.Ri
            end
        else 
            @views model.B[:,:,i] = sqrt.(model.D[:,:,i]) * Mquantities.Uqi * 
            Diagonal(sqrt.(Mquantities.Lambda[1:Mquantities.qi,i] - vec(ones(1,Mquantities.qi)))) * Mquantities.Ri
        end

        #Update D
        Mquantities.C = Array(1.0*I(Squantities.p)); Mquantities.Us = Mquantities.Uqi*(Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi,i])-I(Mquantities.qi)); Mquantities.V = Array(Mquantities.Uqi')
        Mquantities.bvec = zeros(Squantities.p,1)
        #Mquantities.psitilde = Array{Float64,2}(undef,p,1) 
        @inbounds for r in 1:Squantities.p
            @views Mquantities.c = Mquantities.C[1:r,r]
            @views Mquantities.v = Mquantities.V[:,1:r]*Mquantities.c
            @views Mquantities.bvec[r:Squantities.p] = Mquantities.Us[r:Squantities.p,:]*Mquantities.v
            Mquantities.b = Mquantities.bvec[r] + 1
            @views Mquantities.a = Array(Mquantities.v')*Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi, i] .- Mquantities.Lambda[1:Mquantities.qi,i])*Mquantities.v + Array(Mquantities.c')*Mquantities.Vtilde[1:r,1:r,i] * Mquantities.c
            Mquantities.psitilde[r] = max((((Mquantities.a .- Mquantities.b)/Mquantities.b^2 .+ 1)*model.D[r,r,i])[1],Squantities.eta)
            Mquantities.omega = Mquantities.psitilde[r]/model.D[r,r,i] - 1
            if r<Squantities.p && Mquantities.omega!=0
                Mquantities.ratio = Mquantities.omega/(1 + Mquantities.omega*Mquantities.b)
                @views Mquantities.C[1:r, (r+1):Squantities.p] = Mquantities.C[1:r, (r+1):Squantities.p] .- Mquantities.ratio .* Mquantities.c*Array(Mquantities.bvec[(r+1):Squantities.p]')        
            end 
        end
        @views model.D[:,:,i] = Diagonal(Mquantities.psitilde[:,1]) 
    end 
    if Squantities.update_q 
        if Mquantities.qup.q_change
            model.B = Mquantities.qup.B_update
        end 
    end #problem is here...
    #if Mquantities.i_break 
    #    return "failedModel"
    #else 
    #    return model
    #end 
end


function MStep_msmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
    #update = MFAModelM(model.pivec,model.mu,model.B,model.D,model.g,model.q,model.tau,model.logL,model.BIC)
    #p = size(Y,2); n = size(Y,1)
    #if p==2 
    #    q_max = 1    
    #else 
    #    q_max = Int(floor(p + (1 - sqrt(1 + 8*p))/2))
    #end
    Mquantities.n_i = sum(model.tau, dims = 1)
     
    #CM-Step 1: Updating pi and mu_i
    model.pivec = Mquantities.n_i ./ Squantities.n 
    #Squantities.logL_calc!(model, Mquantities, Squantities)
    #println("After updating pivec logL is")
    #println(model.logL)
    #mu_new = Array{Float64,2}(undef,p,g)
    #Mquantities.tau_xi_ij = zeros(n, model.g)
    @inbounds for i in 1:model.g
        Mquantities.tau_xi_ij[:,i] = model.tau[:,i] .* model.w_esteps[:,1,i]
        @views model.mu[:,i] = sum(Squantities.Y .* Mquantities.tau_xi_ij[:,i], dims = 1) ./ sum(Mquantities.tau_xi_ij[:,i]) 
    end 
    #Squantities.logL_calc!(model, Mquantities, Squantities)
    #println("After updating mu logL is")
    #println(model.logL)
    #CM-Step 2: Updating q 
    #Vtilde = @MArray zeros(Float64,p,p,model.g)
    #Lambda = @MMatrix zeros(Float64,q_max,model.g)
    #Lambda_new = Array{Float64,2}(undef,q_max,model.g)
    #eigenvecs =  Array{Any,1}(undef,model.g)
    ecm_failed = false 
    @inbounds for i in 1:model.g
        @views Mquantities.Ymu = Squantities.Y .- model.mu[:,i]' 
        @views Mquantities.Vi = (Array(Mquantities.Ymu') * (Mquantities.Ymu .* Mquantities.tau_xi_ij[:,i]))/Mquantities.n_i[i]
        @views Mquantities.D_is = Diagonal(1 ./ sqrt.(model.D[:,:,i]))
        Mquantities.Vtilde[:,:,i] = Mquantities.D_is*Mquantities.Vi*Mquantities.D_is
        @views Mquantities.V_eig_temp = try
            eigvals(Array(Mquantities.Vtilde[:,:,i]))
        catch e
            model.status = "failed-MInfNan"
            return 
        end
        

        #if any(imag.(V_eig_temp) .!= 0)   
        #    print("uh oh")
        #    print(imag.(V_eig_temp))
        #    print("converting to real number >:(")
        #end 
        #V_eig = real.(V_eig_temp)
         #end 




        if any(imag(Mquantities.V_eig_temp) .> 0)
            println("Complex eigenvalues appeared!!")
            model.status = "failed"
            return
        end 

        if all( Mquantities.V_eig_temp .< 1 )
            ecm_failed = true 
            #println("Complex updating step used")
            #println(Mquantities.V_eig_temp)

        #else
            #println("...")
        end

        Mquantities.eig_order = sortperm(Mquantities.V_eig_temp, rev = true)
        @views Mquantities.Lambda[:,i] = Mquantities.V_eig_temp[Mquantities.eig_order][1:Squantities.ledermann]
        @views Mquantities.eigenvecs[i] = real.(eigvecs(Array(Mquantities.Vtilde[:,:,i]))[:,Mquantities.eig_order])
        if any(real.(Mquantities.Lambda[:,i]) .< 0)
            Mquantities.Lambda[(real.(Mquantities.Lambda[:,i]) .< 0),i] .= 1e-20
            println("Negative eigenvalues appeared!")
        end
    end


    if (!ecm_failed) | (model.name != "MGHFA" )

        #CM-Step 3: Updating B_i, then D_i in same loop.
        Mquantities.qi_break = false 
        @inbounds for i in 1:model.g
            #Update B
            @views Mquantities.qi = sum(Mquantities.Lambda[:,i] .> 1)
            if Mquantities.qi > model.q 
                Mquantities.qi = model.q
            end 

            if Mquantities.qi == 0
                #qi_break = true 
                Mquantities.qi = argmin(cumsum(log.(Mquantities.Lambda[:,i]) .- Mquantities.Lambda[:,i] .+ 1))
                #println("No eigenvalues greater than 1")
            break
            end 
            @views Mquantities.Uqi = Mquantities.eigenvecs[i][:,1:Mquantities.qi]
            @views Mquantities.Ri = I(model.q)[1:Mquantities.qi,:]
            diagM = 
            @views model.B[:,:,i] = sqrt.(model.D[:,:,i]) * Mquantities.Uqi * Diagonal(sqrt.(Complex.(Mquantities.Lambda[1:Mquantities.qi,i]' .- ones(1,Mquantities.qi)))[1,:]) * Mquantities.Ri #The diagonal breaks this... 

            #Update D
            Mquantities.C = Array(1.0*I(Squantities.p)); Mquantities.Us = Mquantities.Uqi*(Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi,i])-I(Mquantities.qi)); Mquantities.V = Array(Mquantities.Uqi')
            Mquantities.bvec = zeros(Squantities.p,1)
            Mquantities.psitilde = Array{Float64,2}(undef,Squantities.p,1) 
            @inbounds for r in 1:Squantities.p
                @views Mquantities.c = Mquantities.C[1:r,r] 
                @views Mquantities.v = Mquantities.V[:,1:r]*Mquantities.c
                @views Mquantities.bvec[r:Squantities.p] = Mquantities.Us[r:Squantities.p,:]*Mquantities.v
                Mquantities.b = Mquantities.bvec[r] + 1
                @views Mquantities.a = Array(Mquantities.v')*Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi, i] .- Mquantities.Lambda[1:Mquantities.qi,i])*Mquantities.v + Array(Mquantities.c')*Mquantities.Vtilde[1:r,1:r,i] * Mquantities.c
                Mquantities.psitilde[r] = max((((Mquantities.a .- Mquantities.b)/Mquantities.b^2 .+ 1)*model.D[r,r,i])[1],Squantities.eta)
                Mquantities.omega = Mquantities.psitilde[r]/model.D[r,r,i] - 1
                if r<Squantities.p && Mquantities.omega!=0
                    Mquantities.ratio = Mquantities.omega/(1 + Mquantities.omega*Mquantities.b)
                    @views Mquantities.C[1:r, (r+1):Squantities.p] = Mquantities.C[1:r, (r+1):Squantities.p] .- Mquantities.ratio .* Mquantities.c*Array(Mquantities.bvec[(r+1):Squantities.p]')        
                end 
            end

            @views model.D[:,:,i] = Diagonal(Mquantities.psitilde[:,1]) 
            
            

            if any(isnan.(model.D[:,:,i]))
                println("Not good...")
            end

        end 
#        old_ll = model.logL
#        Squantities.logL_calc!(model, Mquantities, Squantities)
#        if old_ll > model.logL
#            for i in 1:model.g
#                model.B[:,:,i] = rand(Normal(),Squantities.p,model.q) #copy(Mquantities.best_model.B)
#            end
        #end
        #println("After updating B and D logL is")
        #println(model.logL)
    else 
        logL_mmvmnfa(model, Mquantities, Squantities)
        eu3 = zeros(Squantities.p,Squantities.p,model.g)
        @inbounds for i in 1:model.g
            Ymu = Squantities.Y .- model.mu[:,i]' 
            bi = model.B[:,:,i]' * inv(model.B[:,:,i]*model.B[:,:,i]' + model.D[:,:,i])
            eu1 = bi * ( ( model.w_esteps[:, 1, i] .* Ymu) .- model.beta[:,i]' )
            eu2 = bi * ( model.w_esteps[:, 2, i] .* Ymu  .- model.beta[:,i]')
            @inbounds for j in 1:Squantities.n
                eu3[:,:,j] = model.w_esteps[j, 2, i]*( I(model.q) - bi*model.B[:,:,i] + bi*Ymu[j,:]'*Ymu[j,:]*bi') - bi*(Ymu[j,:]'*model.beta[:,:,i] + model.beta[:,:,i]' * Ymu[j,:])*bi' + model.w_esteps[j, 1, i] * bi * model.beta[:,:,i]' * model.beta[:,:,i] * bi'
            end


            #V = (Array(Ymu') * (Ymu .* model.tau[:,i]))/Mquantities.n_i[i]
            #gamma = inv(real.(model.B[:,:,i]*model.B[:,:,i]' + model.D[:,:,i]))*model.B[:,:,i]
            #omega = I(model.q) - gamma' * model.B[:,:,i]
            #model.B[:,:,i] =  V*gamma*inv(gamma'*V*gamma + omega)
            #model.D[:,:,i] = diagm(diag(V - V*gamma*model.B[:,:,i]'))
        end

    end

    try 
        #CM Step 4: Updating Psi 
        @inbounds for i in 1:model.g
            @views model.Ψ[:,i] = Squantities.msteps(model.tau[:,i], model.w_esteps[:,:,i], Mquantities)
        end
    catch 
        model.status = "failedΨ"
        return
    end
    #Squantities.logL_calc!(model, Mquantities, Squantities)
    #println("After updating psi logL is")
    #println(model.logL)
end

function MStep_mmvmnfa(model::MMVMNFAModel, Mquantities::MMVMNFAMutableQuantities, Squantities::MMVMNFAStaticQuantities)
    #update = MFAModelM(model.pivec,model.mu,model.B,model.D,model.g,model.q,model.tau,model.logL,model.BIC)
    #p = size(Y,2); n = size(Y,1)
    #if p==2 
    #    q_max = 1    
    #else 
    #    q_max = Int(floor(p + (1 - sqrt(1 + 8*p))/2))
    #end
    Mquantities.n_i = sum(model.tau, dims = 1)
     
    #CM-Step 1: Updating pi and mu_i
    model.pivec = Mquantities.n_i ./ Squantities.n 
    #mu_new = Array{Float64,2}(undef,p,g)
    #tau_rho_xi_ij = zeros(n, model.g); tau_rho_ij = zeros(n, model.g)
    #tr_ij = zeros(n, model.g); tx_ij = zeros(n, model.g)
    @inbounds for i in 1:model.g
        @views Mquantities.xi_bar_i = sum(model.tau[:,i] .* model.w_esteps[:,1,i])/sum(model.tau[:,i])
        @views Mquantities.rho_bar_i = sum(model.tau[:,i] .* model.w_esteps[:,2,i])/sum(model.tau[:,i])
        @views Mquantities.tau_rho_xi_ij[:,i] = model.tau[:,i] .* (Mquantities.xi_bar_i .* model.w_esteps[:,2,i] .- 1)
        @views Mquantities.tau_rho_ij[:,i] = model.tau[:,i] .* (Mquantities.rho_bar_i .- model.w_esteps[:,2,i]) 
        @views Mquantities.tr_ij[:,i] = model.tau[:,i] .* model.w_esteps[:,2,i]
        @views Mquantities.tx_ij[:,i] = model.tau[:,i] .* model.w_esteps[:,1,i]
        @views model.mu[:,i] = sum(Squantities.Y .* Mquantities.tau_rho_xi_ij[:,i], dims = 1) ./ sum(Mquantities.tau_rho_xi_ij[:,i]) 
        @views model.beta[:,i] = sum(Squantities.Y .* Mquantities.tau_rho_ij[:,i], dims = 1) ./ sum(Mquantities.tau_rho_xi_ij[:,i]) 
        #if any(sum(Y .== model.mu[:,i]', dims = 2) == p)
        #    println("Data point has dominated and become the mean of a component.")
        #    break 
        #end
    end 

    #CM-Step 2: Updating q 
    #Vtilde = @MArray zeros(Float64,p,p,model.g)
    #Lambda = @MMatrix zeros(Float64,q_max,model.g)
    #Lambda_new = Array{Float64,2}(undef,q_max,model.g)
    #eigenvecs =  Array{Any,1}(undef,model.g)
    Mquantities.ecm_failed = false
        @inbounds for i in 1:model.g
            Mquantities.beta_matrix = reshape(repeat(model.beta[:,i],inner=1,outer=Squantities.n), Squantities.p, Squantities.n) |> permutedims
            @views Mquantities.Ymu = Squantities.Y .- model.mu[:,i]' 
            @views Mquantities.Vi = (Array(Mquantities.Ymu') * (Mquantities.Ymu .* Mquantities.tr_ij[:,i]) - 
                        Array(Mquantities.Ymu') * (Mquantities.beta_matrix .* model.tau[:,i]) - 
                        Array(Mquantities.beta_matrix') * (Mquantities.Ymu .* model.tau[:,i]) + 
                        Array(Mquantities.beta_matrix') * (Mquantities.beta_matrix .* Mquantities.tx_ij[:,i]))/Mquantities.n_i[i]
            @views Mquantities.D_is = Diagonal(1 ./ sqrt.(model.D[:,:,i]))
            Mquantities.Vtilde[:,:,i] = Mquantities.D_is*Mquantities.Vi*Mquantities.D_is
            #@views Mquantities.V_eig_temp = eigvals(Array(Mquantities.Vtilde[:,:,i]))
            #if any(imag.(V_eig_temp) .!= 0)   
            #    print("uh oh")
            #    print(imag.(V_eig_temp))
            #    print("converting to real number >:(")
            #end 
            @views Mquantities.V_eig_temp = try
                eigvals(Array(Mquantities.Vtilde[:,:,i]))
            catch e
                model.status = "failed-MInfNan"
                return 
            end

            if any(imag(Mquantities.V_eig_temp) .> 0)
                println("Complex eigenvalues appeared!!")
                model.status = "failed"
                return
            end 

            if all( Mquantities.V_eig_temp .< 1 )
                Mquantities.ecm_failed = true 
                #println("Complex updating step used")
                #println(Mquantities.V_eig_temp)

            #else
                #println("...")
            end

            #V_eig = real.(V_eig_temp)

            Mquantities.eig_order = sortperm(Mquantities.V_eig_temp, rev = true)
            
            @views Mquantities.Lambda[:,i] = Mquantities.V_eig_temp[Mquantities.eig_order][1:Squantities.ledermann]
            @views Mquantities.eigenvecs[i] = eigvecs(Array(Mquantities.Vtilde[:,:,i]))[:,Mquantities.eig_order]
            if any(Mquantities.Lambda[:,i] .< 0)
                println("Negative eigenvalues appeared!")
                Mquantities.Lambda[(Mquantities.Lambda[:,i] .< 0),i] .= 1e-20
            end
        end

        if (!Mquantities.ecm_failed) | (model.name != "MGHFA") #true
        #println("ECM update used")
        #CM-Step 3: Updating B_i, then D_i in same loop.
        Mquantities.qi_break = false 
        @inbounds for i in 1:model.g
            #Update B
            @views Mquantities.qi = sum(Mquantities.Lambda[:,i] .> 1)
            if Mquantities.qi > model.q 
                Mquantities.qi = model.q
            end 

            if Mquantities.qi == 0
                Mquantities.qi = argmin(cumsum(log.(Mquantities.Lambda[:,i]) .- Mquantities.Lambda[:,i] .+ 1))
                #qi_break = true 
                #println("No eigenvalues greater than 1")
            #break
            end 
            @views Mquantities.Uqi = Mquantities.eigenvecs[i][:,1:Mquantities.qi]
            @views Mquantities.Ri = I(model.q)[1:Mquantities.qi,:]
            @views model.B[:,:,i] = sqrt.(model.D[:,:,i]) * Mquantities.Uqi * Diagonal(sqrt.(Complex.(Mquantities.Lambda[1:Mquantities.qi,i:i]' - ones(1,Mquantities.qi)))[1,:]) * Mquantities.Ri

            #Update D
            Mquantities.C = Array(1.0*I(Squantities.p)); Mquantities.Us = Mquantities.Uqi*(Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi,i])-I(Mquantities.qi)); Mquantities.V = Array(Mquantities.Uqi')
            Mquantities.bvec = zeros(Squantities.p,1)
            Mquantities.psitilde = Array{Float64,2}(undef,Squantities.p,1) 
            #Mquantities.bvec = zeros(p,1)
        # Mquantities.psitilde = Array{Float64,2}(undef,p,1) #Debugging up to here ... 
            @inbounds for r in 1:Squantities.p
                @views Mquantities.c = Mquantities.C[1:r,r]
                @views Mquantities.v = Mquantities.V[:,1:r]*Mquantities.c
                @views Mquantities.bvec[r:Squantities.p] = Mquantities.Us[r:Squantities.p,:]*Mquantities.v
                Mquantities.b = Mquantities.bvec[r] + 1
                @views Mquantities.a = Array(Mquantities.v')*Diagonal(1 ./ Mquantities.Lambda[1:Mquantities.qi, i] .- Mquantities.Lambda[1:Mquantities.qi,i])*Mquantities.v + Array(Mquantities.c')*Mquantities.Vtilde[1:r,1:r,i] * Mquantities.c
                Mquantities.psitilde[r] = max((((Mquantities.a .- Mquantities.b)/Mquantities.b^2 .+ 1)*model.D[r,r,i])[1],Squantities.eta)
                Mquantities.omega = Mquantities.psitilde[r]/model.D[r,r,i] - 1
                if r<Squantities.p && Mquantities.omega!=0
                    Mquantities.ratio = Mquantities.omega/(1 + Mquantities.omega*Mquantities.b)
                    @views Mquantities.C[1:r, (r+1):Squantities.p] = Mquantities.C[1:r, (r+1):Squantities.p] .- Mquantities.ratio .* Mquantities.c*Array(Mquantities.bvec[(r+1):Squantities.p]')        
                end 
            end
            @views model.D[:,:,i] = Diagonal(Mquantities.psitilde[:,1]) 
        end 
    else
        #println("AECM update used")
        logL_mmvmnfa(model, Mquantities, Squantities)
        #println(model.logL)
        eu3 = zeros(model.q,model.q,Squantities.n)
        @inbounds for i in 1:model.g
            Ymu = Squantities.Y .- model.mu[:,i]' 
            bi = model.B[:,:,i]' * inv(model.B[:,:,i]*model.B[:,:,i]' + model.D[:,:,i])

            eu1 = (bi * ( Ymu .- model.w_esteps[:, 1, i].*model.beta[:,i]' )')'
            eu2 = (bi * ( model.w_esteps[:, 2, i] .* Ymu  .- model.beta[:,i]')')'
            @inbounds for j in 1:Squantities.n
                eu3[:,:,j] = model.w_esteps[j, 2, i]*( I(model.q) - bi*model.B[:,:,i] + bi*Ymu[j:j,:]'  *Ymu[j:j,:]*bi') - bi*(Ymu[j:j,:]'*model.beta[:,i:i]' + model.beta[:,i:i] * Ymu[j:j,:])*bi' + model.w_esteps[j, 1, i] * bi * model.beta[:,i:i] * model.beta[:,i:i]' * bi'
            end

            #*inv(sum([model.tau[j,i] .* eu3[:,:,j] for j in 1:Squantities.n],dims = 3))

            model.B[:,:,i] = sum([model.tau[j,i] * (Ymu[j:j,:]'*eu2[j:j,:] .- model.beta[:,i:i]*eu1[j:j,:]) for j in 1:Squantities.n]  ) * inv(sum([model.tau[j,i] .* eu3[:,:,j] for j in 1:Squantities.n], dims = 1)[1])



            logL_mmvmnfa(model, Mquantities, Squantities)

            #update for B not quite right?? 

            #println(model.logL)
            model.D[:,:,i] = 1/(sum(model.tau[:,i]))*diagm(diag(sum(
            [ model.tau[j,i] * (
            model.w_esteps[j,2,i]*Ymu[j:j,:]'*Ymu[j:j,:] - 2*model.beta[:,i:i]*Ymu[j:j,:] + model.w_esteps[j,1,i]*model.beta[:,i:i]*model.beta[:,i:i]' - 2*Ymu[j:j,:]'*eu2[j:j,:]*model.B[:,:,i]' + 2*model.beta[:,i:i]*eu1[j:j,:]*model.B[:,:,i]' + model.B[:,:,i]*eu3[:,:,j]*model.B[:,:,i]'
            )
            for j in 1:Squantities.n]
            ,dims=1)[1]))
            #logL_mmvmnfa(model, Mquantities, Squantities)
            #println(model.logL)
            #eu3[:,:,j] for j in 1,dims = 1)
            #V = (Array(Ymu') * (Ymu .* model.tau[:,i]))/Mquantities.n_i[i]
            #gamma = inv(real.(model.B[:,:,i]*model.B[:,:,i]' + model.D[:,:,i]))*model.B[:,:,i]
            #omega = I(model.q) - gamma' * model.B[:,:,i]
            #model.B[:,:,i] =  V*gamma*inv(gamma'*V*gamma + omega)
            #model.D[:,:,i] = diagm(diag(V - V*gamma*model.B[:,:,i]'))
        end

    end
    try 
        #CM Step 4: Updating Psi 
        @inbounds for i in 1:model.g
            @views model.Ψ[:,i] = Squantities.msteps(model.tau[:,i], model.w_esteps[:,:,i], Mquantities)
        end
    catch 
        model.status = "failedΨ"
        return
    end
end