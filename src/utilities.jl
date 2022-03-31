#####
##### utilities
#####

"""
$(SIGNATURES)

Return `(F, is_valid_F)` where

1. `F` is a form (usually a factorization) of the argument that supports `F \\ r` for
   vectors `r`,

2. `is_valid_F` is a boolean indicating whether `F` can be used in this form (eg `false` for
   a singular factorization).
"""
function _factorize(J::AbstractMatrix)
    LU = lu(J; check = false)
    LU, issuccess(LU)
end

function _factorize(J::Diagonal)
    J, all(x -> x > 0, diag(J))
end

"""
$(SIGNATURES)

Ellipsoidal norm ``\\| x \\|_B = x'Bx``.
"""
ellipsoidal_norm(x, ::UniformScaling) = norm(x, 2)
