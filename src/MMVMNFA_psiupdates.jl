#Mixture of t Factor Analyzers
function mtfa_msteps(tau, estep_estimates, Mquantities,lower_nu = 0.1,upper_nu = 200)
  #browser()
  #tau_ij  <- estep_estimates[,1]
    #xi_ij = estep_estimates[:,1]
    #zeta_ij = estep_estimates[:,2]
    #MAKE THIS USE GLOBAL VARIABLES 
    Mquantities.mq.fnu = function f_nu_i(nu_i)
      #log(nu_i/2) + 1 - digamma(nu_i/2) + sum(tau_ij*(zeta_ij - xi_ij))/sum(tau_ij)
      return - (sum(tau)*(0.5*nu_i*log(0.5*nu_i) - log(gamma(0.5*nu_i))) + 0.5*nu_i*sum(tau .* (estep_estimates[:,2] .- estep_estimates[:,1])))
    end

    #function f_nu_i(nu_i)
      #log(nu_i/2) + 1 - digamma(nu_i/2) + sum(tau_ij*(zeta_ij - xi_ij))/sum(tau_ij)
    #  return log.(0.5*nu_i) .+ 1 .- digamma.(0.5*nu_i) .+ sum(tau .* (zeta_ij .- xi_ij))/sum(tau) 
    #end

    #return [find_zero(f_nu_i, (lower_nu, upper_nu), Bisection())]

    Mquantities.mq.optim_min = optimize(Mquantities.mq.fnu, lower_nu, upper_nu)

    if !Optim.converged(Mquantities.mq.optim_min)
      print("Error estimating nu")
      return nothing
    end

    return [Optim.minimizer(Mquantities.mq.optim_min)]
end  

#Mixture of Slash Factor Analyzers 
function mslfa_msteps(tau, estep_estimates,Mquantities)
    return  [-sum(tau)/sum(tau.*estep_estimates[:,2])]
end

#Mixture of Contaminated Normal Factor Analyzers
function mcnfa_msteps(tau,estep_estimates,Mquantities)
    #tau_ij  = estep_estimates[:,1]
    #kappa_ij = estep_estimates[:,3]
    #delta_ij = estep_estimates[:,4]
    #p = estep_estimates[1,6] #This entry is just a column of p's 
    #nu_i = 
    #gamma_i = 
  return [sum(tau .* estep_estimates[:,2])/sum(tau); estep_estimates[1,4]*sum(tau .* estep_estimates[:,2])/sum(tau .* estep_estimates[:,2] .* estep_estimates[:,3])]
end
#Mixture of Contaminated Normal Factor Analyzers

function mghfa_msteps(tau, estep_estimates, Mquantities, lower_lam = -100, upper_lam = 100, lower_omega = 1e-4, upper_omega = 500)
    #browser()
    #tau_ij  <- estep_estimates[,1]
    #tau_ij  <- estep_estimates[,1]
    #xi_ij <- estep_estimates[,1]
    #rho_ij <- estep_estimates[,2]
    ##zeta_ij <- estep_estimates[,3]
    #omega_k <- estep_estimates[1,4]
    #lambda_k <- estep_estimates[1,5] 

  Mquantities.mq.zeta_bar = sum(estep_estimates[:,3].*tau)/sum(tau)
  Mquantities.mq.xi_bar = sum(estep_estimates[:,1].*tau)/sum(tau)
  Mquantities.mq.rho_bar = sum(estep_estimates[:,2].*tau)/sum(tau)
  
  Mquantities.mq.flam = function f_lam_i(lam_i)
      #log(nu_i/2) + 1 - digamma(nu_i/2) + sum(tau_ij*(zeta_ij - xi_ij))/sum(tau_ij)
      return - ( Mquantities.mq.zeta_bar*lam_i - log(besselk(lam_i,estep_estimates[1,4])))
  end

  Mquantities.mq.fomega = function f_omega_i(omega_i)
    #log(nu_i/2) + 1 - digamma(nu_i/2) + sum(tau_ij*(zeta_ij - xi_ij))/sum(tau_ij)
    return  log(besselk(estep_estimates[1,5],omega_i)) + 0.5*omega_i*( Mquantities.mq.xi_bar+ Mquantities.mq.rho_bar)
  end

  Mquantities.mq.min_lam = optimize(Mquantities.mq.flam, lower_lam, upper_lam)
  
  Mquantities.mq.min_omega = optimize(Mquantities.mq.fomega, lower_omega, upper_omega)


  if !Optim.converged(Mquantities.mq.min_lam) | !Optim.converged(Mquantities.mq.min_omega)
    print("Error estimating scaling density parameters")
    return nothing
  end

  return [Optim.minimizer(Mquantities.mq.min_omega);Optim.minimizer(Mquantities.mq.min_lam)]
end  
  
function mbsfa_msteps(tau,estep_estimates,Mquantities)
#  #estep_order: tau, xi, rho
  #tau_ij  <- estep_estimates[,1]
  #xi_ij = estep_estimates[:,1] 
  #rho_ij = estep_estimates[:,2] 
  Mquantities.mq.xi_bar = sum(estep_estimates[:,1] .* tau)/sum(tau)
  Mquantities.mq.rho_bar = sum(estep_estimates[:,2] .* tau)/sum(tau)
  return [sqrt(Mquantities.mq.xi_bar + Mquantities.mq.rho_bar -2)]
end

function mlfa_msteps(tau,estep_estimates,Mquantities)
  #estep_order: tau, xi
    #au_ij  <- estep_estimates[,1]
    #xi_ij <- estep_estimates[:,1]  
    Mquantities.mq.xi_bar = sum(estep_estimates[:,1] .* tau)/sum(tau)
    #if(((xi_bar-1)^2 + 8*xi_bar)<0){browser()}
    return [(-(Mquantities.mq.xi_bar-1) + sqrt((Mquantities.mq.xi_bar-1)^2 + 8*Mquantities.mq.xi_bar))/(2*Mquantities.mq.xi_bar)]
end
  