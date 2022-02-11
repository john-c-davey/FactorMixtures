mutable struct MMVMNFAModel 
    name::String
    pivec::Matrix{Float64}
    mu::Matrix{Float64}
    beta::Matrix{Float64} #Stochastic mean component
    B::Union{Array{Float64},Array{ComplexF64}}
    D::Array{Float64}
    Ψ::Union{Matrix{Float64},Nothing} #Parameters of the scaling density function
    g::Int64
    q::Int64
    tau::Matrix{Float64}
    w_esteps::Union{Array{Float64},Nothing} #Esteps for the scaling density function
    logL::Float64
    BIC::Float64
    status::String
    niter::Int64
    logL_history::Vector{Float64}
end

mutable struct BUpdateQuantities
    Lambda_cs::Matrix{Float64}
    n_param::Vector{Int64}
    q_scores::Matrix{Float64}
    q_up::Int64
    q_change::Bool
    B_update::Array{ComplexF64}
end

mutable struct MtFAQuantities 
    fnu::Any
    optim_min::Any #Define the optim min class here... and union with nothing
end

mutable struct MSLFAQuantities 
    integrand::Any
    int_res::Float64
    c::Vector{Float64} #possible vector?? 
    integrand_estep::Any
    int_res_estep::Vector{Float64}
end

mutable struct MCNFAQuantities 
    d1::Vector{Float64}
    d2::Vector{Float64}
    kappa_ij::Vector{Float64}
end

mutable struct MGHFAQuantities 
    zeta_bar::Float64 
    xi_bar::Float64 
    rho_bar::Float64
    flam::Any  
    fomega::Any
    min_lam::Any #Define the optim min class here... and union with nothing
    min_omega::Any #Define the optim min class here... and union with nothing
    delta_b::Float64
    omega::Float64
    lambda::Float64
    t1::Vector{Float64}
    t2::Vector{Float64}
    integrand::Any
    t3::Vector{Float64}
end

mutable struct MBSFAQuantities 
    xi_bar::Float64 
    rho_bar::Float64
    a_is::Float64
    delta_b::Float64
    Theta::Vector{Float64}
    Lambda::Float64
    a::Float64
    b1::Vector{Float64}
    b2::Vector{Float64}
end

mutable struct MLFAQuantities 
    xi_bar::Float64 
    a::Float64
    delta_b::Float64
    numerator::Vector{Float64}
    py::Vector{Float64}
    rat::Vector{Float64}
    b1::Vector{Float64}
    b2::Vector{Float64}
    b3::Vector{Float64}
end

mutable struct MMVMNFAMutableQuantities
    BIC_best::Float64
    best_model::Union{Nothing, MMVMNFAModel}
    g::Int64
    q::Int64
    k_starts::Array{Float64,2}
    r_starts::Array{Float64,2}
    starts::Array{Float64,2}
    ind::Int64
    init_para_error::Union{String,Bool}
    indices::Any
    Si::Matrix{Float64}
    Di_sqrt::Matrix{Float64}
    inv_Di_sqrt::Matrix{Float64}
    lambda::Any
    eig_order::Any
    lambda_vals::Any
    sigma2::Float64
    maxinit::Int64
    sigma::Matrix{Float64}
    delta::Vector{Float64}
    Fji::Matrix{Float64}
    Fjmax::Matrix{Float64}
    n_i::Matrix{Float64}
    Vtilde::Array{Float64}
    Lambda::Matrix{Float64}
    Ymu::Array{Float64, 2}
    Vi::Array{Float64, 2}
    D_is::Any
    V_eig_temp::Any
    V_eig::Any
    qi_break::Bool
    qi::Int64
    Uqi::Any
    Ri::Any
    C::Any
    Us::Any
    V::Any
    bvec::Any
    psitilde::Array{Float64,2}
    c::Any
    v::Any
    b::Any
    a::Any
    omega::Any
    ratio::Any
    eigenvecs::Array{Any,1}
    logL_old::Float64
    d_model::Int64
    tau_rho_xi_ij::Matrix{Float64}
    tau_rho_ij::Matrix{Float64}
    tr_ij::Matrix{Float64}
    tx_ij::Matrix{Float64}
    xi_bar_i::Float64
    rho_bar_i::Float64
    beta_matrix::Matrix{Float64}
    mq::Union{Nothing,MtFAQuantities,MSLFAQuantities,MCNFAQuantities,MGHFAQuantities,MBSFAQuantities,MLFAQuantities}
    tau_xi_ij::Matrix{Float64}
    qup::Union{Nothing,BUpdateQuantities}
end

struct MMVMNFAStaticQuantities
    Y::Array{Float64, 2}
    gmin::Int64
    gmax::Int64
    qmin::Int64
    qmax::Int64
    eta::Float64
    itmax::Int64
    nkmeans::Int64
    nrandom::Int64
    psi_initial::Union{Nothing, Vector{Float64}}
    update_q::Bool 
    tol::Float64
    conv_measure::String
    n::Int64
    p::Int64
    n_it::Int64
    n_comb::Int64
    combs::Union{Nothing, Matrix{Tuple{Int64, Int64}}}
    ecm_fit!::Any
    logfY!::Any
    esteps!::Any
    msteps::Any
    n_esteps::Union{Int64, Nothing}
    logL_calc!::Any
    Mstep_calc!::Any
end

