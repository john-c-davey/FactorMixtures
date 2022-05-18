#Density of MtFA
function logfY_MtFA(model, Mquantities, Squantities, index)
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]') + model.D[:,:,index]
    Mquantities.Fji[:,index] = logpdf(MvTDist(model.Ψ[1,index],Vector(model.mu[:,index]), Matrix(Mquantities.sigma)),Squantities.Y')
    #return logpdf(MvTDist(model.Ψ[1,index],Vector(model.mu[:,index]), Matrix(Mquantities.sigma)),Squantities.Y') 
  #return(mvtnorm::dmvt(data, delta = mu, sigma = Re(B%*%t(B)+D), df = w_params))
end

#Density of MSLFA
function logfY_MSLFA(model, Mquantities, Squantities, index)
    #n = size(data)[1]
    #Mquantities.mq.int_container = zeros(Squantities.n,1)
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]') + model.D[:,:,index]
    for j in 1:Squantities.n
        Mquantities.mq.integrand = function integrand(w)
            w^(model.Ψ[index] - 1) * pdf(MvNormal(Vector(model.mu[:,index]), Matrix(Mquantities.sigma)/w ), Squantities.Y[j,:] )
        end 
        Mquantities.mq.int_res = log(model.Ψ[index]*quadgk( Mquantities.mq.integrand,0,1,rtol=1e-10)[1])
        if Mquantities.mq.int_res == -Inf 
            Mquantities.Fji[j,index] = -500
        else
            Mquantities.Fji[j,index] = Mquantities.mq.int_res
        end
    end
end

#Density of MCNFA
function logfY_MCNFA(model, Mquantities, Squantities, index)
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]') + model.D[:,:,index]
    Mquantities.Fji[:,index] =  log.(
      model.Ψ[1,index] .* pdf(MvNormal(Vector(model.mu[:,index]), Matrix(Mquantities.sigma)/model.Ψ[2,index] ), Squantities.Y')  .+
      (1-model.Ψ[1,index]).*pdf(MvNormal(Vector(model.mu[:,index]), Matrix(Mquantities.sigma) ), Squantities.Y' ) 
      )
end

function logd_GH(data,chi,psi,lambda,B, mu, D, beta)
#  #Density of Generalised Hyperbolic distribution.
    n,p = size(data)
    sig_i = real(B*B')+D
    delta = Distances.colwise(SqMahalanobis(inv(sig_i)), Array(data'), mu)
    #delta = mahalanobis(data, mu, inv(sig_i), inverted = FALSE)
    delta_b = beta'*inv(sig_i)*beta
    if chi != 0
        return 0.5*(lambda - 0.5*p) .* (log.(chi .+ delta) .- log(psi + delta_b)) .+
           0.5*lambda*(log(psi)-log(chi)) .+ log.( besselk.(lambda-0.5*p , sqrt.( (psi+delta_b) .* (chi .+ delta) ))) .-
           0.5*p*log(2*pi) .- 0.5 * log(det(sig_i)) .- log(besselk(lambda, sqrt(psi*chi)))  .+ 
           (data - (reshape(repeat(mu,inner=1,outer=n), p, n) |> permutedims) )*inv(sig_i)*beta   
    else 
    #Check this to make sure I've calculated the limiting case density correctly...!
        return 0.5*(lambda - 0.5*p) .* (log.(chi .+ delta) .- log.(psi .+ delta_b)) .+
        lambda*(log(psi)) .+ log.(besselk.(lambda-0.5*p, sqrt.((psi.+delta_b) .* (chi .+ delta) ))) .-
        0.5*p*log(2*pi) .- 0.5 * log(det(sig_i)) .- (lambda-1)*log(2) .- lgamma(lambda) .+ 
        (data - (reshape(repeat(mu,inner=1,outer=n), p, n) |> permutedims) )*inv(sig_i)*beta
        #if(any(is.nan(temp))){browser()}
        # temp
    end
end

#Density of GHFA
function logfY_MGHFA(model, Mquantities, Squantities, index)
    Mquantities.Fji[:,index] = logd_GH(Squantities.Y,model.Ψ[1,index],model.Ψ[1,index],model.Ψ[2,index],model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index])
end
    
  
function logfY_MBSFA(model, Mquantities, Squantities, index)
    Mquantities.mq.a_is = 1/(model.Ψ[index]^2)
    Mquantities.Fji[:,index] =log.(0.5*exp.(logd_GH(Squantities.Y,Mquantities.mq.a_is,Mquantities.mq.a_is,0.5,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index])) .+ 
               0.5*exp.(logd_GH(Squantities.Y,Mquantities.mq.a_is,Mquantities.mq.a_is,-0.5,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index]))) 
end

function logfY_MLFA(model, Mquantities, Squantities, index)
    Mquantities.mq.a = model.Ψ[index]
    Mquantities.Fji[:,index] = log.(Mquantities.mq.a/(1+Mquantities.mq.a)*exp.(logd_GH(Squantities.Y,0,2*Mquantities.mq.a,1,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index])) .+ 
               1/(1+Mquantities.mq.a)*exp.(logd_GH(Squantities.Y,0,2*Mquantities.mq.a,2,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index])))
end

