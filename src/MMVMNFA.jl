#####
#To DO:
#Add support for the whole MMVMNFA model family
#Stop using static arrays for tau and anything which scales up with n 
#Change the initialisation procedure??
#Catch complex eigenvalues during initialisation
#####

using LinearAlgebra, Distributions, Random, DataFrames, DelimitedFiles, Plots, Distances, Clustering, ProgressMeter, StaticArrays

mutable struct MMVMNModel 
    name::String
    pivec::Matrix{Float64}
    mu::Matrix{Float64}
    beta::Matrix{Float64} #Stochastic mean component
    B::Array{Float64}
    D::Array{Float64}
    Ψ::Union{Matrix{Float64},Nothing} #Parameters of the scaling density function
    g::Int64
    q::Int64
    tau::Matrix{Float64}
    logL::Float64
    BIC::Float64
end

"
This is the main fitting function for 3 (nested) classes of model.
"
function MFA_ECM(Y::Array{Float64, 2}, gmin::Int64, gmax::Int64 , qmin::Int64 , eta::Float64  ,itmax::Int64 , nkmeans::Int64 , nrandom::Int64 ,tol = 1e-05, conv_measure = "diff")
    
    n = size(Y,1); p = size(Y,2); 
    if p==2
        qmax = 1
    else 
        qmax = Int(floor(p + (1-sqrt(1+8*p))/2))
    end 

    if qmax < qmin 
        error("Invalid choice of q_min.")
    end 


    n_it = Int((gmax-gmin+1)*(qmax-qmin+1)*(nkmeans+nrandom))
    prog = Progress(n_it)
  
    BIC_best = Inf
    best_model = missing
  
    n_comb = (gmax-gmin+1)*(qmax-qmin+1)
    combs = reshape(collect(Iterators.product(gmin:gmax, qmin:qmax)),(n_comb,1)); 
    #Threads.@threads 
    Threads.@threads for i in 1:n_comb
        g = combs[i][1]; q = combs[i][2]
            initial_partitions = start_clust(Y, g, nkmeans,nrandom)
            maxinit = size(initial_partitions,2)
            if ismissing(maxinit)
                maxinit = 1
            end 

            init_model_para = "placeholder"
            estd_model = "placeholder"
            for ii in 1:maxinit 
                
            init_model_para = init_est_para_mfa(Y, g, q, initial_partitions[:, ii])
            if init_model_para == "failedModel"
                # Failed 
            else 
                estd_model = est_mfa(init_model_para, Y, itmax, tol, conv_measure, eta)

                if estd_model == "failedModel"
                    # Failed
                else 
                    estd_q = q
                    d_model = (g - 1) + 2*g * p + g * (p * estd_q - estd_q * (estd_q - 1)/2)
                    estd_model.BIC = (-2 * estd_model.logL + d_model * log(n))
                    if (estd_model.BIC < BIC_best)
                        best_model = estd_model; BIC_best = estd_model.BIC; #model["BIC"] = BIC_best
                    end 
                end 
            end
           
            next!(prog)
            end
    end
  
  return best_model
end

"
Creates a matrix of starting values of dimension n x (n_kmeans + n_random). These are initial clusterings from which we calculate initial parameter values. 
"
function start_clust(Y::Matrix{Float64}, g::Int64, nkmeans::Int64, nrandom::Int64) 
    n = size(Y,1)
    
    init_clust = missing 
    k_starts = Array{Float64,2}(undef,n,nkmeans)
    if !ismissing(nkmeans) && nkmeans > 0
        @inbounds for i in 1:nkmeans
            k_starts[:, i] = assignments(kmeans(Array(Y'), g))
        end
    end
  
  if !ismissing(nrandom) && nrandom > 0
    r_starts = reshape(rand(1:g, n * nrandom), n, nrandom)
  end
  nc = ifelse(ismissing(init_clust), 0, 1 ) #dummy here, will need to fix later

  starts = Array{Float64,2}(undef,n,nkmeans + nrandom)
  ind = 1

  if nkmeans > 0 
    starts[:, ind:(ind - 1 + nkmeans)] = k_starts
    ind = ind + nkmeans
  end

  if nrandom > 0
    starts[:, ind:(ind - 1 + nrandom)] = r_starts
    ind = ind + nrandom
  end

  return starts
end

"
Generates initial parameter values for a given start (i.e. a given initial clustering)
"
function init_est_para_mfa(Y::Matrix{Float64}, g::Int64, q::Int64 , start::Union{Vector{Float64},Matrix{Float64}}) 

    p = size(Y,2)
    n = size(Y,1)
    
    B = @MArray zeros(Float64,p,q,g) 
    pivec = @MArray zeros(Float64,1,g)
    mu = @MArray zeros(Float64,p,g)
    D = @MArray zeros(Float64,p,p,g)
    error = "no"
    @inbounds for i in 1:g 
        @views indices = findall(start .== i)[:,1]
        pivec[i] = length(indices)/n
        @views mu[:, i] = mean(Y[indices,:], dims = 1)
        @views Si = cov(Y[indices, :])
        D[:,:,i] = Diagonal(Si)
        @views Di_sqrt = sqrt.(D[:,:,i])
        inv_Di_sqrt = Diagonal(1 ./ diag(Di_sqrt))
       
        lambda = try 
                    eigen(inv_Di_sqrt * Si * inv_Di_sqrt)
                 catch e
                    error = "yes"
                    break
                 end 
        eig_order = sortperm(lambda.values, rev = true) #Can get complex eigenvalues here...!
        lambda_vals = lambda.values[eig_order]
        
        if q == p
            sigma2 = 0
        else 
            @views sigma2 = mean(lambda_vals[(q + 1):p])
        end

        if q == 1
            @views B[:,:,i] = Di_sqrt * lambda.vectors[:, eig_order[1:q]] * (lambda_vals[1:q] .- sigma2)
        else 
            @views B[:,:,i] = Di_sqrt * lambda.vectors[:, eig_order[1:q]] * Diagonal((lambda_vals[1:q] .- sigma2))
        end                                              
    end

    if error == "yes"
        return "failedModel"
    end

    return MFAModel(pivec,mu,mu,B,D,Nothing,g,q,pivec,0,1)
end





# Add this struct throughout as function output to be reference quickly. 
#Then include a temporary struct which can be used to store variable values until the end of each M-step. 
#struct MFAModel 
#    pivec::Matrix{Float64}
#    mu::Matrix{Float64}
#    B::Array{Float64}
 #   D::Array{Float64}
 #   g::Int64
 #   q::Int64
 #   tau::Matrix{Float64}
 #   logL::Float64
 #   BIC::Float64
#end

function chol_inv(x)
  C <- chol(x)
  inv_x <- chol2inv(C)
  return(inv_x)
end

function MStep_amfa(Y::Matrix{Float64},model::MFAModel,eta::Float64)
    update = MFAModelM(model.pivec,model.mu,model.B,model.D,model.g,model.q,model.tau,model.logL,model.BIC)
    p = size(Y,2); n = size(Y,1)
    if p==2 
        q_max = 1    
    else 
        q_max = Int(floor(p + (1 - sqrt(1 + 8*p))/2))
    end
    n_i = sum(model.tau, dims = 1)
     
    #CM-Step 1: Updating pi and mu_i
    update.pivec = n_i ./ n 
    #mu_new = Array{Float64,2}(undef,p,g)
    @inbounds for i in 1:model.g
        @views update.mu[:,i] = sum(Y .* model.tau[:,i], dims = 1) ./ sum(model.tau[:,i]) 
    end 

    #CM-Step 2: Updating q 
    Vtilde = @MArray zeros(Float64,p,p,model.g)
    Lambda = @MMatrix zeros(Float64,q_max,model.g)
    #Lambda_new = Array{Float64,2}(undef,q_max,model.g)
    eigenvecs =  Array{Any,1}(undef,model.g)
    @inbounds for i in 1:model.g
        @views Ymu = Y .- update.mu[:,i]' 
        @views Vi = (Array(Ymu') * (Ymu .* model.tau[:,i]))/n_i[i]
        @views D_is = Diagonal(1 ./ sqrt.(model.D[:,:,i]))
        Vtilde[:,:,i] = D_is*Vi*D_is
        @views V_eig_temp = eigvals(Array(Vtilde[:,:,i]))
        #if any(imag.(V_eig_temp) .!= 0)   
        #    print("uh oh")
        #    print(imag.(V_eig_temp))
        #    print("converting to real number >:(")
        #end 
        V_eig = real.(V_eig_temp)
        eig_order = sortperm(V_eig, rev = true)
        @views Lambda[:,i] = V_eig[eig_order][1:q_max]
        @views eigenvecs[i] = real.(eigvecs(Array(Vtilde[:,:,i]))[:,eig_order])
        if any(Lambda[:,i] .< 0)
            Lambda[(Lambda[:,i] .< 0),i] .= 1e-20
        end
    end

    #CM-Step 3: Updating B_i, then D_i in same loop.
    qi_break = false 
    @inbounds for i in 1:model.g
        #Update B
        @views qi = sum(Lambda[:,i] .> 1)
        if qi > model.q 
            qi = model.q
        end 

        if qi == 0
            qi_break = true 
           break
        end 
        @views Uqi = eigenvecs[i][:,1:qi]
        @views Ri = I(model.q)[1:qi,:]
        @views update.B[:,:,i] = sqrt.(model.D[:,:,i]) * Uqi * Diagonal(sqrt.(Lambda[1:qi,i] - ones(1,qi))) * Ri

        #Update D
        C = Array(1.0*I(p)); Us = Uqi*(Diagonal(1 ./ Lambda[1:qi,i])-I(qi)); V = Array(Uqi')
        bvec = zeros(p,1)
        psitilde = Array{Float64,2}(undef,p,1) #Debugging up to here ... 
        @inbounds for r in 1:p
            @views c = C[1:r,r]
            @views v = V[:,1:r]*c
            @views bvec[r:p] = Us[r:p,:]*v
            b = bvec[r] + 1
            @views a = Array(v')*Diagonal(1 ./ Lambda[1:qi, i] .- Lambda[1:qi,i])*v + Array(c')*Vtilde[1:r,1:r,i] * c
            psitilde[r] = max((((a .- b)/b^2 .+ 1)*update.D[r,r,i])[1],eta)
            omega = psitilde[r]/update.D[r,r,i] - 1
            if r<p && omega!=0
                ratio = omega/(1 + omega*b)
                @views C[1:r, (r+1):p] = C[1:r, (r+1):p] .- ratio .* c*Array(bvec[(r+1):p]')        
            end 
        end
        @views update.D[:,:,i] = Diagonal(psitilde[:,1]) 
    end 
    if qi_break 
        return "failedModel"
    else 
        return MFAModel(update.pivec,update.mu,update.B,update.D,model.g,model.q,model.tau,model.logL,model.BIC)
    end 
end

function logL_tau_mfa(Y::Matrix{Float64},model::MFAModel)

    p = size(Y,2)
    n = size(Y,1)
    Fji = @MMatrix zeros(Float64,n,model.g)
    @inbounds for i in 1:model.g 
        
        @views inv_D = Diagonal(1 ./ model.D[:,:,i])
        @views B_inv_D = model.B[:,:,i] .* diag(inv_D)
        @views inv_O = inv( I(model.q) + B_inv_D' * model.B[:,:,i] )
        inv_S = inv_D - B_inv_D * inv_O * B_inv_D'
        @views logdetS = sum(log.(diag(model.D[:,:,i]))) - log(det(inv_O))

        @views mahal_dist = Distances.colwise(SqMahalanobis(inv_S), Array(Y'), model.mu[:,i])

        Fji[:, i] = -0.5 .* mahal_dist .- (p/2) .* log(2 * pi) .- 0.5 .* logdetS

    end

    Fji = Fji .+ log.(model.pivec)
    Fjmax = maximum(Fji, dims = 2)
    Fji = Fji .- Fjmax 
    logL = sum(Fjmax) + sum(log.(sum(exp.(Fji), dims = 2)))
    Fji = exp.(Fji)
    tau = Fji ./ sum(Fji, dims = 2)
    #model.logL = loglike 
    #model.tau = tau
    #return Dict([("loglike", loglike), ("tau", tau)])
    return MFAModel(model.pivec,model.mu,model.B,model.D,model.g,model.q,tau,logL,model.BIC) #model 
end











