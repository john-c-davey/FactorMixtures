#Mixture of t Factor Analyzers
 function mtfa_esteps(model, Mquantities, Squantities, index)
  #p = size(data)[2]
  Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
  Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
  #delta = mahalanobis(data, mu_i, real(B_i%*%t(B_i) + D_i), inverted = FALSE)
  #xi_ij = 
  #zeta_ij = 
  model.w_esteps[:,:,index] = hcat((model.Ψ[index] + Squantities.p) ./ (model.Ψ[index] .+ Mquantities.delta), digamma((model.Ψ[index] + Squantities.p)/2) .- log.((model.Ψ[index] .+ Mquantities.delta)/2))
 end

#Need to tacking the slash... why does it have to be so annoying :<

#Mixture of Slash Factor Analyzers 
function mslfa_esteps(model, Mquantities, Squantities, index)
  #p = ncol(data); n = nrow(data)
    #n,p = size(data)
    #sig_i = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]') + model.D[:,:,index]
    Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
    Mquantities.mq.c = cdf.(Gamma.(model.Ψ[index]+Squantities.p/2, 1 ./(0.5*Mquantities.delta) ), 1)
    #xi_ij = 2*(model.Ψ[index]+Squantities.p/2).* cdf.(Gamma.(model.Ψ[index]+Squantities.p/2 + 1, 1 ./(0.5*Mquantities.delta) ), 1) ./ (Mquantities.delta.* c) #Need to fix this 
    #int_res = zeros(n,1)                                             
    @inbounds for i in 1:Squantities.n
        Mquantities.mq.integrand_estep = function integrand(u)
            u^(model.Ψ[index] + Squantities.p/2 - 1)*exp(-0.5*Mquantities.delta[i]*u)*log(u)
        end
        Mquantities.mq.int_res_estep[i] = quadgk(Mquantities.mq.integrand_estep,0,1,rtol = 1e-10)[1]
    end 
    #zeta_ij = int_res.*((0.5*delta).^(model.Ψ[index]+p/2))./(gamma(model.Ψ[index]+p/2) .* c)
  model.w_esteps[:,:,index] = hcat(2*(model.Ψ[index]+Squantities.p/2).* cdf.(Gamma.(model.Ψ[index]+Squantities.p/2 + 1, 1 ./(0.5*Mquantities.delta) ), 1) ./ (Mquantities.delta.* Mquantities.mq.c),
  Mquantities.mq.int_res_estep.*((0.5*Mquantities.delta).^(model.Ψ[index]+Squantities.p/2))./(gamma(model.Ψ[index]+Squantities.p/2) .* Mquantities.mq.c)) #shape scale cdf(Gamma(model.Ψ[index]+p/2 + 1, 1./(0.5*delta) ), 1)
end

#Mixture of Contaminated Normal Factor Analyzers
function mcnfa_esteps(model, Mquantities, Squantities, index)
    #n,p = size(data)
    #sig_i = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
    Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
    Mquantities.mq.d1 = pdf(MvNormal(Vector(model.mu[:,index]), Matrix(Mquantities.sigma) ), Squantities.Y' )
    Mquantities.mq.d2 = pdf(MvNormal(Vector(model.mu[:,index]), Matrix(Mquantities.sigma)/model.Ψ[2,index] ), Squantities.Y' )
    Mquantities.mq.kappa_ij = 1 .- ((1-model.Ψ[1,index])*Mquantities.mq.d1) ./ ((1-model.Ψ[1,index])*Mquantities.mq.d1 + model.Ψ[1,index]*Mquantities.mq.d2)
  #kappa_ij = 1 .- rho_ij
    #xi_ij = (1 .- kappa_ij) .+ model.Ψ[2,index]*kappa_ij
    model.w_esteps[:,:,index] = hcat((1 .- Mquantities.mq.kappa_ij) .+ model.Ψ[2,index]*Mquantities.mq.kappa_ij, Mquantities.mq.kappa_ij, 
    Mquantities.delta, repeat([Squantities.p],Squantities.n)) 
end


#Mixture of GH Factor Analyzers
function mghfa_esteps(model, Mquantities, Squantities, index)
  #c() wrappers needed here
    #n,p = size(data) 
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
    Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
    Mquantities.mq.delta_b = model.beta[:,index]'*inv(Mquantities.sigma)*model.beta[:,index]
    Mquantities.mq.omega = model.Ψ[1,index]
    Mquantities.mq.lambda = model.Ψ[2,index]
    Mquantities.mq.t1 = (Mquantities.mq.omega .+ Mquantities.delta)/(Mquantities.mq.omega + Mquantities.mq.delta_b)
    Mquantities.mq.t2 = besselk.(Mquantities.mq.lambda- 0.5*Squantities.p + 1, sqrt.((Mquantities.mq.omega + Mquantities.mq.delta_b)*(Mquantities.mq.omega .+ Mquantities.delta)))./
    besselk.(Mquantities.mq.lambda - 0.5*Squantities.p, sqrt.((Mquantities.mq.omega + Mquantities.mq.delta_b)*(Mquantities.mq.omega .+ Mquantities.delta)))
    #t3 = ones(n,1)
    @inbounds for j in 1:Squantities.n 
      Mquantities.mq.integrand = function t2point5(t)
            log(besselk(t, sqrt((Mquantities.mq.omega + Mquantities.mq.delta_b)*(Mquantities.mq.omega + Mquantities.delta[j]))))
        end
        #Mquantities.init_para_error = false
        try 
          Mquantities.mq.t3[j] = central_fdm(8, 1)(Mquantities.mq.integrand, Mquantities.mq.lambda - 0.5*Squantities.p) #(, lambda - 0.5*p),silent=TRUE) #derivative term here. derivative effectively zero if other terms are bad...!
        catch 
          #Mquantities.init_para_error = true
          model.status = "failedE"
        end 
    #if(class(temp)=="try-error"){t3[j]<- 0}else{t3[j] <- temp_deriv}
    end
    #xi_ij <- c(sqrt(t1))*c(t2) # -(2*lambda - p)./(omega .+ delta) + sqrt.(1./t1) .* t2
    #rho_ij <- -1*rep(2*lambda - p,n)/(rep(omega,n) + delta) + sqrt(1/t1)*t2 
    #zeta_ij <- log(sqrt(t1)) + t3 
    model.w_esteps[:,:,index] = hcat(sqrt.(Mquantities.mq.t1) .* Mquantities.mq.t2, -(2*Mquantities.mq.lambda - Squantities.p)./(Mquantities.mq.omega .+ Mquantities.delta) .+ sqrt.(1 ./Mquantities.mq.t1) .* Mquantities.mq.t2 , log.(sqrt.(Mquantities.mq.t1)) .+ Mquantities.mq.t3, repeat([Mquantities.mq.omega],Squantities.n), repeat([Mquantities.mq.lambda],Squantities.n) )
end

#Mixture of BS Factor Analyzers
function mbsfa_esteps(model, Mquantities, Squantities, index)
  
    #n,p = size(data) # <- nrow(data); p <- ncol(data)
    Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
    Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
    Mquantities.mq.delta_b = model.beta[:,index]'*inv(Mquantities.sigma)*model.beta[:,index]
    Mquantities.mq.a = model.Ψ[index]
    Mquantities.mq.Theta = Mquantities.delta .+ Mquantities.mq.a^-2
    Mquantities.mq.Lambda = Mquantities.mq.delta_b + Mquantities.mq.a^-2
  
    Mquantities.mq.b1 = besselk.((1-Squantities.p)/2, sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))
    Mquantities.mq.b2 = besselk.((1+Squantities.p)/2, sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))
  
    #xi_ij = 0.5*(Theta./Lambda)^(0.5) .* (besselk(1 + (1-p)/2, sqrt.(Theta.*Lambda))./b1 + besselk(1 - (1+p)/2, sqrt.(Theta.*Lambda))./b2  )
 
    #rho_ij = 0.5*(Theta./Lambda)^(-0.5) .* (besselk(-1 + (1-p)/2,sqrt.(Theta.*Lambda))./b1 + besselk(-1 - (1+p)/2, sqrt.(Theta.*Lambda))./b2  ) 
  
    model.w_esteps[:,:,index] = hcat(0.5*(Mquantities.mq.Theta/Mquantities.mq.Lambda).^(0.5) .* (besselk.(1 + (1-Squantities.p)/2, sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))./Mquantities.mq.b1 + besselk.(1 - (1+Squantities.p)/2, sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))./Mquantities.mq.b2  ),
                0.5*(Mquantities.mq.Theta/Mquantities.mq.Lambda).^(-0.5) .* (besselk.(-1 + (1-Squantities.p)/2,sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))./Mquantities.mq.b1 + besselk.(-1 - (1+Squantities.p)/2, sqrt.(Mquantities.mq.Theta*Mquantities.mq.Lambda))./Mquantities.mq.b2  ))
end 

#Mixture of Lindley Factor Analyzers
function mlfa_esteps(model, Mquantities, Squantities, index)
#  #c() wrappers needed here
  #n,p = size(data)
  #sig_i = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
  #delta = Distances.colwise(SqMahalanobis(inv(sig_i)), Array(data'), model.mu[:,index])
  Mquantities.sigma = real(model.B[:,:,index]*model.B[:,:,index]')+model.D[:,:,index]
  Mquantities.delta = Distances.colwise(SqMahalanobis(inv(Mquantities.sigma)), Array(Squantities.Y'), model.mu[:,index])
  #Mquantities.mq.delta_b = model.beta[:,index]'*inv(Mquantities.sigma)*model.beta[:,index]
  Mquantities.mq.a = model.Ψ[index]
  Mquantities.mq.delta_b = model.beta[:,index]'*inv(Mquantities.sigma)*model.beta[:,index] + 2*Mquantities.mq.a
  
  Mquantities.mq.numerator = Mquantities.mq.a*exp.(logd_GH(Squantities.Y,0,2*Mquantities.mq.a,1,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index]))
  Mquantities.mq.py = Mquantities.mq.numerator./(Mquantities.mq.numerator .+ exp.(logd_GH(Squantities.Y,0,2*Mquantities.mq.a,2,model.B[:,:,index], model.mu[:,index], model.D[:,:,index], model.beta[:,index])))
  Mquantities.mq.rat = Mquantities.delta./Mquantities.mq.delta_b

  Mquantities.mq.b1 = besselk.(1 - Squantities.p/2, sqrt.(Mquantities.delta.*Mquantities.mq.delta_b))
  Mquantities.mq.b2 = besselk.(2 - Squantities.p/2, sqrt.(Mquantities.delta.*Mquantities.mq.delta_b))
  Mquantities.mq.b3 = besselk.(3 - Squantities.p/2, sqrt.(Mquantities.delta.*Mquantities.mq.delta_b))
  
  #xi_ij = rat.^(1/2).*(py.*b2./b1 + (1 .- py).*b3./b2)
  #rho_ij = rat.^(-1/2).*(py*besselk(- p/2, sqrt.(delta*delta_b))./b1 + (1 .- py).*b1./b2)

  model.w_esteps[:,:,index] = hcat(Mquantities.mq.rat.^(1/2).*(Mquantities.mq.py .* Mquantities.mq.b2 ./ Mquantities.mq.b1 + (1 .- Mquantities.mq.py).*Mquantities.mq.b3 ./Mquantities.mq.b2),
  Mquantities.mq.rat .^(-1/2).*(Mquantities.mq.py.*besselk.(-Squantities.p/2, sqrt.(Mquantities.delta*Mquantities.mq.delta_b))./Mquantities.mq.b1 + (1 .- Mquantities.mq.py).*Mquantities.mq.b1./Mquantities.mq.b2))
end 