"
This is the main fitting function for 3 (nested) classes of model.
"
function MFA_ECM(Y::Array{Float64, 2}, name::String, gmin::Int64, gmax::Int64 , qmin::Int64 ,qmax::Int64, eta::Float64 , itmax::Int64 , nkmeans::Int64 , nrandom::Int64, psi_initial::Union{Nothing, Vector{Float64}}=nothing, update_q::Bool = false, tol::Float64 = 1e-05, conv_measure::String = "diff")
    
    @assert gmin <= gmax "gmin must be less than or equal to gmax"
    @assert gmin >= 1 "there must be at least one component"
    @assert eta > 0.0 "eta, the smallest possible entry of any error variance matrix, must be positive"
    @assert itmax >= 1 "at least one EM iteration must be permitted"
    @assert nkmeans>=0&&nrandom>=0 "both the number of initialisations from kmeans and the number of random initialisations must be positive"
    @assert nkmeans>=1||nrandom>=1 "at least on initialisation is required"
    @assert tol > 0.0 "tol, the tolerance of the EM convergence criterion, must be positive"

    n,p = size(Y)
    led = Int(floor(p + (1-sqrt(1+8*p))/2))

    @assert qmax <= led "qmax must obey the Ledermann bound"

    if name == "MFA"
        ecm_fit! = est_mfa
        logfY! = nothing; esteps! = nothing; msteps = nothing; n_esteps = nothing
        logL_calc! = logL_mfa; Mstep_calc! = MStep_mfa
        mq = nothing; extra_params = 0
    elseif name == "MtFA"
        ecm_fit! = est_msmnfa
        logfY! = logfY_MtFA; esteps! = mtfa_esteps; msteps = mtfa_msteps; n_esteps = 2  
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_msmnfa
        mq = MtFAQuantities(nothing,nothing); extra_params = 1
    elseif name == "MSLFA"
        ecm_fit! = est_msmnfa
        logfY! = logfY_MSLFA; esteps! = mslfa_esteps; msteps = mslfa_msteps; n_esteps = 2 
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_msmnfa
        mq = MSLFAQuantities(nothing,0.0,Vector{Float64}(undef,n),nothing,Vector{Float64}(undef,n)); extra_params = 1
    elseif name == "MCNFA"
        ecm_fit! = est_msmnfa
        logfY! = logfY_MCNFA; esteps! = mcnfa_esteps; msteps = mcnfa_msteps; n_esteps = 4   
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_msmnfa
        mq = MCNFAQuantities(Vector{Float64}(undef,n),Vector{Float64}(undef,n),Vector{Float64}(undef,n)); extra_params = 2
    elseif name == "MGHFA"
        ecm_fit! = est_mmvmnfa
        logfY! = logfY_MGHFA; esteps! = mghfa_esteps; msteps = mghfa_msteps; n_esteps = 5  
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_mmvmnfa  
        mq = MGHFAQuantities(0.0,0.0,0.0,nothing,nothing,nothing,nothing,0.0,0.0,0.0,Vector{Float64}(undef,n),Vector{Float64}(undef,n),nothing,Vector{Float64}(undef,n)); extra_params = (p+2)
    elseif name == "MBSFA"
        ecm_fit! = est_mmvmnfa
        logfY! = logfY_MBSFA; esteps! = mbsfa_esteps; msteps = mbsfa_msteps; n_esteps = 2 
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_mmvmnfa  
        mq = MBSFAQuantities(0.0,0.0,0.0,0.0,Vector{Float64}(undef,n),0.0,0.0,Vector{Float64}(undef,n),Vector{Float64}(undef,n)); extra_params = (p+1)  
    elseif name == "MLFA"  
        ecm_fit! = est_mmvmnfa
        logfY! = logfY_MLFA; esteps! = mlfa_esteps; msteps = mlfa_msteps; n_esteps = 2    
        logL_calc! = logL_mmvmnfa; Mstep_calc! = MStep_mmvmnfa 
        mq = MLFAQuantities(0.0,0.0,0.0,Vector{Float64}(undef,n),Vector{Float64}(undef,n),Vector{Float64}(undef,n),Vector{Float64}(undef,n),Vector{Float64}(undef,n),Vector{Float64}(undef,n)); extra_params = (p+1)
    elseif name == "MFA_AECM"
        ecm_fit! = est_mfa_aecm
        logfY! = nothing; esteps! = nothing; msteps = nothing; n_esteps = nothing
        logL_calc! = logL_mfa; Mstep_calc! = Mstep_calc_aecm
        mq = nothing; extra_params = 0
    end 
    
    if update_q
        qup = BUpdateQuantities(Matrix{Float64}(undef,1,1),Vector{Int64}(undef,1),Matrix{Float64}(undef,1,1),0,false,Array{ComplexF64}(undef))
        qmin = 1
    else    
        qup = nothing
    end

    #Mquantities = MMVMNFAMutableQuantities(Inf, nothing)
    if !update_q
        Squantities = MMVMNFAStaticQuantities(Y, gmin, gmax, qmin, qmax, eta, itmax, nkmeans, nrandom, psi_initial, update_q, tol, conv_measure, n , p, 
                                            Int((gmax-gmin+1)*(qmax-qmin+1)*(nkmeans+nrandom)), 
                                            Int((gmax-gmin+1)*(qmax-qmin+1)),
                                            reshape(collect(Iterators.product(gmin:gmax, qmin:qmax)),(Int((gmax-gmin+1)*(qmax-qmin+1)),1)), 
                                            ecm_fit!, logfY!, esteps!, msteps, n_esteps, logL_calc!, Mstep_calc!, extra_params, led)
    else 
        Squantities = MMVMNFAStaticQuantities(Y, gmin, gmax, qmin, qmax, eta, itmax, nkmeans, nrandom, psi_initial, update_q, tol, conv_measure, n , p, 
                                            Int((gmax-gmin+1)*(nkmeans+nrandom)), 
                                            Int(gmax-gmin+1),
                                            reshape(collect(Iterators.product(gmin:gmax, qmin:qmin)),(Int(gmax-gmin+1),1)), 
                                            ecm_fit!, logfY!, esteps!, msteps, n_esteps, logL_calc!, Mstep_calc!, extra_params,led)
    end

    
    #if Squantities.qmax < Squantities.qmin 
    #    error("Invalid choice of q_min.")
    #end 
    if (gmin==1)
        if !update_q
            prog = Progress(Squantities.n_it - (qmax-qmin+1)*(nkmeans+nrandom-1))
        else 
            prog = Progress(Squantities.n_it - (nkmeans+nrandom-1))
        end
    else
        prog = Progress(Squantities.n_it)
    end
    #logfY, esteps, msteps, n_esteps::Int64,

    BIC_best = Inf 
    best_model = nothing 
    #model_results = Matrix{Float64}(undef, Squantities.n_comb*(nrandom+nkmeans), 5)
    #k = 1

    
   #Threads.@threads 
   for i in 1:Squantities.n_comb
        println("New Model")
        #initialise a model and a quantity struct here
        model = MMVMNFAModel(name,zeros(Float64,1,Squantities.combs[i][1]), zeros(Float64,Squantities.p,Squantities.combs[i][1]), zeros(Float64,Squantities.p,Squantities.combs[i][1]), zeros(ComplexF64,p,Squantities.combs[i][2],Squantities.combs[i][1]), 
        zeros(Float64,Squantities.p,Squantities.p,Squantities.combs[i][1]), nothing, Squantities.combs[i][1], Squantities.combs[i][2], zeros(Float64,1,Squantities.combs[i][1]), nothing, 0, 1, "Completed",0,Vector{Float64}(undef, Squantities.itmax+1), Vector{String}(), repeat([1],Squantities.n), 0.0) 
        # g = Squantities.combs[i][1]; q = Squantities.combs[i][2]
         
        Mquantities = MMVMNFAMutableQuantities(-Inf, nothing, model.g, model.q, Array{Float64,2}(undef,Squantities.n,Squantities.nkmeans), 
        Array{Float64,2}(undef,Squantities.n,Squantities.nrandom), Array{Float64,2}(undef,Squantities.n,Squantities.nkmeans + Squantities.nrandom),1,
        "no", nothing, zeros(Squantities.p,Squantities.p),zeros(Squantities.p,Squantities.p),zeros(Squantities.p,Squantities.p), nothing, nothing, nothing, 0,
        1, zeros(1,1),[0.0], zeros(Float64,Squantities.n,model.g) , zeros(Float64,Squantities.n,1), zeros(Float64,Squantities.ledermann,model.g),
        zeros(Float64,Squantities.p,Squantities.p,model.g),
        zeros(Float64,Squantities.ledermann,model.g),
        #Array{Any,1}(undef,model.g),
        Squantities.Y, zeros(Float64,Squantities.ledermann,model.g), zeros(Float64,Squantities.ledermann,model.g),
        zeros(Float64,Squantities.ledermann,model.g), zeros(Float64,Squantities.ledermann,model.g), false, 0.0,
        zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),Array{Float64,2}(undef,Squantities.p,1),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g),zeros(Float64,Squantities.ledermann,model.g), Array{Any,1}(undef,model.g), Inf, Int(0), 
        zeros(Squantities.n, model.g), zeros(Squantities.n, model.g), zeros(Squantities.n, model.g), zeros(Squantities.n, model.g), 0.0, 0.0, zeros(Squantities.n, model.g),mq, zeros(Squantities.n, model.g),qup, false, Vector{Int64}[],Vector{Int64}[])
    #Lambda = @MMatrix zeros(Float64,q_max,model.g))

        start_clust!(Mquantities, Squantities)

        Mquantities.maxinit = size(Mquantities.starts,2)
        if (ismissing(Mquantities.maxinit)) | (model.g == 1)
            Mquantities.maxinit = 1
        end 

        #init_model_para = "placeholder"
        #estd_model = "placeholder"
        for ii in 1:Mquantities.maxinit 
                
            init_est_para_mfa!(model, Mquantities, Squantities, ii)

            if Mquantities.init_para_error == "yes"
                # Failed, add in code here eventually to track the successes and failures...
            else 
                #estd_model = ecm_fit(init_model_para, Y, logfY, esteps, msteps, n_esteps, psi_initial, itmax, tol, conv_measure, eta, update_q)
                loop_time = @elapsed ecm_fit!(model, Mquantities, Squantities)
                if model == "failedModel"
                    # Failed
                else 
                    #estd_q = q
                    model.time = loop_time
                    Mquantities.d_model = (model.g - 1) + 2*model.g * Squantities.p + model.g * (Squantities.p * model.q - model.q * (model.q - 1)/2) + model.g*(Squantities.extra_params) 
                    model.BIC = (-2 * model.logL + Mquantities.d_model * log(Squantities.n))
                    if (model.BIC < BIC_best)
                        best_model = deepcopy(model); BIC_best = deepcopy(model.BIC); #model["BIC"] = BIC_best
                    end 
                    #model_results[k,:] = [float(model.g) float(model.q) model.BIC model.logL model.logL_history[1]]
                    #k = k + 1
                end 
            end
           
            next!(prog)
        end
    end
  best_model.clustering = last.(Tuple.(argmax(best_model.tau, dims = 2)))
  return best_model
end

"
Creates a matrix of starting values of dimension n x (n_kmeans + n_random). These are initial clusterings from which we calculate initial parameter values. 
"
function start_clust!(Mquantities, Squantities) 
    #n = size(Y,1)
    
    #init_clust = missing 
    #k_starts = Array{Float64,2}(undef,n,nkmeans)
    #if !ismissing(nkmeans) && nkmeans > 0
    @inbounds for i in 1:Squantities.nkmeans
        Mquantities.k_starts[:, i] = assignments(kmeans(Array(Squantities.Y'), Mquantities.g)) 
    end
    #end
  
  #if !ismissing(nrandom) && nrandom > 0
    Mquantities.r_starts = reshape(rand(1:Mquantities.g, Squantities.n * Squantities.nrandom), Squantities.n, Squantities.nrandom)
  #end
    #nc = ifelse(ismissing(init_clust), 0, 1 ) #dummy here, will need to fix later

    #starts = Array{Float64,2}(undef,n,nkmeans + nrandom)
    Mquantities.ind = 1

    if Squantities.nkmeans > 0 
        Mquantities.starts[:, Mquantities.ind:(Mquantities.ind - 1 + Squantities.nkmeans)] = Mquantities.k_starts
        Mquantities.ind = Mquantities.ind + Squantities.nkmeans
    end

    if Squantities.nrandom > 0
        Mquantities.starts[:, Mquantities.ind:(Mquantities.ind - 1 + Squantities.nrandom)] = Mquantities.r_starts
        #Mquantities.ind = Mquantities.ind + nrandom
    end

end

"
Generates initial parameter values for a given start (i.e. a given initial clustering)
"
function init_est_para_mfa!(model, Mquantities, Squantities, index) 

    #p = size(Y,2)
    #n = size(Y,1)
    
    #B = zeros(ComplexF64,p,q,g) 
    #pivec = @MArray zeros(Float64,1,g)
    #mu = @MArray zeros(Float64,p,g)
    #D = @MArray zeros(Float64,p,p,g)
    #error = "no"
    @inbounds for i in 1:model.g 
        @views Mquantities.indices = findall(Mquantities.starts[:,index] .== i)[:,1]
        model.pivec[i] = length(Mquantities.indices)/Squantities.n
        @views model.mu[:, i] = mean(Squantities.Y[Mquantities.indices,:], dims = 1)
        @views Mquantities.Si = cov(Squantities.Y[Mquantities.indices, :])
        model.D[:,:,i] = Diagonal(Mquantities.Si)
        @views Mquantities.Di_sqrt = sqrt.(model.D[:,:,i])
        Mquantities.inv_Di_sqrt = Diagonal(1 ./ diag(Mquantities.Di_sqrt))
       
        Mquantities.lambda = try 
                                eigen(Mquantities.inv_Di_sqrt * Mquantities.Si * Mquantities.inv_Di_sqrt)
                            catch e
                                Mquantities.init_para_error = "yes"
                                #println("Initialisation error")
                                break
                            end 
        if any(imag.(Mquantities.lambda.values) .> 0)
            println("Complex eigenvalues appeared!!")
            model.status = Mquantities.init_para_error = "yes"
            break
        end 
        Mquantities.eig_order = sortperm(Mquantities.lambda.values, rev = true) #Can get complex eigenvalues here...!
        Mquantities.lambda_vals = Mquantities.lambda.values[Mquantities.eig_order]
        
        if model.q == Squantities.p
            #Mquantities.sigma2 = 0
            Mquantities.sigma2 = 1
        else 
            @views Mquantities.sigma2 = mean(Mquantities.lambda_vals[(Mquantities.q + 1):Squantities.p])
            #Mquantities.sigma2 = 1
        end
        if Squantities.update_q
            model.B = Array{Float64}(undef, Squantities.p, model.q, model.g) #In case the last model had changed q.
        end
        if model.q == 1
            @views model.B[:,:,i] = Complex.(Mquantities.Di_sqrt * Mquantities.lambda.vectors[:, Mquantities.eig_order[1:model.q]] * (Mquantities.lambda_vals[1:model.q] .- Mquantities.sigma2))
        else 
            @views model.B[:,:,i] = Complex.(Mquantities.Di_sqrt * Mquantities.lambda.vectors[:, Mquantities.eig_order[1:model.q]] * Diagonal((Mquantities.lambda_vals[1:model.q] .- Mquantities.sigma2)))
        end                                              
    end

    #if Mquantities.init_para_error == "yes"
    #    return "failedModel"
    #end

    #return MMVMNFAModel(modelName,pivec,mu,mu,B,D,nothing,g,q,pivec,nothing,0,1,"Completed")
end



