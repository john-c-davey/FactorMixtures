module FactorMixtures

# Write your package code here.
using LinearAlgebra, Distributions, Random, DataFrames, Distances, Clustering, ProgressMeter, StaticArrays, QuadGK, FiniteDifferences, SpecialFunctions, Optim, Roots

export MFA_ECM, start_clust!, init_est_para_mfa!, mtfa_esteps, mtfa_msteps

include("MMVMNFA_model_structs.jl")
include("MMVMNFA_drivers.jl")
include("MMVMNFA_Esteps.jl")
include("MMVMNFA_fitters.jl")
include("MMVMNFA_logden.jl")
include("MMVMNFA_loglikes.jl")
include("MMVMNFA_Msteps_complex.jl")
include("MMVMNFA_psiupdates.jl")
include("mfa_aecm.jl")

end
