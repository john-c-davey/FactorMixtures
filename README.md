# FactorMixtures
Julia package for fitting the MFA model and models from the MMVMNFA model family in the case where the hyperparameters g (the number of components) and q (the number of factors per component) are unknown.

## Example

The following simple example shows how FactorMixtures can be used to fit an MFA model and a MGHFA model to an example dataset. 

```julia
using DelimitedFiles
test_dataset,names = readdlm(download("https://raw.githubusercontent.com/john-c-davey/Misc/main/mtfa_test_dataset.csv"), ',', Float64, '\n', header = true)
_test_mfa = MFA_ECM(test_dataset, "MFA", 1, 2, 1, 1,0.5e-2, 200, 1, 1) 
_test_mghfa = MFA_ECM(test_dataset, "MGHFA", 1, 2, 1, 1,0.5e-2, 200, 1, 1, [1.0,1.0]) 
```

[![Build Status](https://github.com/john-c-davey/FactorMixtures.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/john-c-davey/FactorMixtures.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/john-c-davey/FactorMixtures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/john-c-davey/FactorMixtures.jl)
# FactorMixtures
