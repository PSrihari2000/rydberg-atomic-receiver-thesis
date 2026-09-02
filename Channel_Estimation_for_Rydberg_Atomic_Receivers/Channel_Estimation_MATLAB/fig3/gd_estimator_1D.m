function Ghat = gd_estimator_1D(Y, absB, Z, S)
% GD_ESTIMATOR_1D  Exact minimizer of the GD objective, Eq.(13)-(15), for the
% 1D array (no rank projection needed, Sec.III-A).
%
% Eq.13's objective L(G) = ||Y - |B| - Re(Z.*(GS))||_F^2 is an UNCONSTRAINED
% CONVEX QUADRATIC in G (real/imaginary parts) -- it has no local optima, and
% it decomposes into I INDEPENDENT per-antenna real least-squares problems,
% since row i of the residual only depends on row i of G. Each is solved here
% in closed form (MATLAB backslash), which is exactly what iterative gradient
% descent (Eq.14-15) converges to -- this sidesteps having to guess a step
% size / iteration count / tolerance the paper never specifies, and removes
% "not fully converged" as a possible source of error entirely.
%
% Derivation (per antenna i, real/imag split g_i = x_i + j*y_i, K real x_i,
% K real y_i): for pilot slot p, Re(g_i^T*(S(:,p)*Z(i,p))) is LINEAR in
% [x_i;y_i] via Re((x+jy)^T(a+jc)) = x^T*a - y^T*c. Stacking over p gives a
% standard real P x 2K design matrix D_i = [Re(V).', -Imag(V).'], where
% V = S.*Z(i,:) (K x P); solving D_i*[x_i;y_i] = Y(i,:).'-absB(i,:).' by
% least squares (D_i \ t_i) gives the exact global minimizer for row i.
%
% Inputs:
%   Y     (I x P real)    - observed signal, Eq.(11)/(12) LHS
%   absB  (I x P real)    - |b_{i,p}|
%   Z     (I x P complex) - e^{-j*angle(b_{i,p})}
%   S     (K x P complex) - pilot matrix
%
% Output:
%   Ghat (I x K complex) - estimated channel (exact minimizer of Eq.13)

[I, ~] = size(Y);
K = size(S,1);

Ghat = zeros(I, K);
for i = 1:I
    V = S .* Z(i,:);              % K x P, V(:,p) = S(:,p)*Z(i,p)
    D = [real(V).', -imag(V).'];  % P x 2K
    t = Y(i,:).' - absB(i,:).';   % P x 1

    sol = D \ t;                  % 2K x 1 real LS solve, exact global minimizer
    Ghat(i,:) = (sol(1:K) + 1i*sol(K+1:2*K)).';
end

end
