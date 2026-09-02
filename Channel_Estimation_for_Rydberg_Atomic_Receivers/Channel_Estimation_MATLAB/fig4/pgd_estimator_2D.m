function Ghat_flat = pgd_estimator_2D(Y_flat, absB_flat, Z_flat, S, I1, I2, Lk, maxIter, reltol)
% PGD_ESTIMATOR_2D  Projected gradient descent channel estimator, Algorithm 1
% (Eq.23-27), for the 2D array. Unlike fig3's 1D GD, this genuinely needs to
% be iterative: the per-user rank constraint (Sec.III-B) makes the feasible
% set non-convex-in-general (though the projection onto it, per user, is
% exact via Eckart-Young-Mirsky), so there's no single closed-form solve.
%
% Internally works in two orientations:
%  - "flat" (I1*I2 x K), matching fig3's G convention, used for the gradient
%    step -- the residual/gradient formula is then IDENTICAL in form to
%    fig3's 1D case (this was verified by transposing Eq.24 into this
%    convention, see fig3/gd_estimator_1D.m's docstring for the base derivation).
%  - "G_(3)" (K x I1*I2), the paper's own mode-3-unfolded convention, needed
%    ONLY for the projection step since each user's rank constraint is a
%    per-ROW constraint on G_(3) (Eq.25-27).
%
% Inputs:
%   Y_flat, absB_flat, Z_flat (I1*I2 x P) - observed data, |b|, phase term
%   S       (K x P complex)   - pilot matrix
%   I1, I2                    - array dimensions
%   Lk      (K x 1)           - per-user rank constraint (TRUE path count,
%                                given as a known input per Algorithm 1)
%   maxIter, reltol           - PGD stopping controls (not given numerically
%                                by the paper); reltol is relative
%                                (||G^{t+1}-G^t||_F/||G^t||_F) since |G|~1e8
%
% Output:
%   Ghat_flat (I1*I2 x K complex) - estimated channel, flat convention

IJ = I1*I2;
K  = size(S,1);

% init: G0_(3) entries iid CN(0,0.1), Sec.V
Gflat = sqrt(0.1/2)*(randn(IJ,K) + 1i*randn(IJ,K));

eta0 = 1/(2*norm(S,2)^2);   % conservative fixed step from S's spectral norm (as fig3)

resid_fun = @(Gf) Y_flat - absB_flat - real((Gf*S).*Z_flat);
loss_fun  = @(Gf) norm(resid_fun(Gf), 'fro')^2;

curLoss = loss_fun(Gflat);
for t = 1:maxIter
    % ---- gradient step, Eq.(23)-(24) (transposed into "flat" convention) ----
    r = resid_fun(Gflat);
    grad_flat = -2*(r .* conj(Z_flat)) * S';     % I1*I2 x K

    step = eta0;
    Mflat = Gflat - step*grad_flat;
    newLoss = loss_fun(Mflat);
    bt = 0;
    while newLoss > curLoss && bt < 20
        step = step/2;
        Mflat = Gflat - step*grad_flat;
        newLoss = loss_fun(Mflat);
        bt = bt + 1;
    end

    % ---- projection step, Eq.(25)-(27): per-user SVD truncation ----
    M3 = Mflat.';           % K x I1*I2, "G_(3)" convention
    X3 = zeros(K, IJ);
    for k = 1:K
        Mk = reshape(M3(k,:), I1, I2);      % back to native I1 x I2 shape
        [U, Sig, V] = svd(Mk);
        r_k = Lk(k);
        Uk = U(:,1:r_k); Sigk = Sig(1:r_k,1:r_k); Vk = V(:,1:r_k);
        Xk = Uk * Sigk * Vk';               % Eckart-Young-Mirsky optimal rank-Lk approx
        X3(k,:) = reshape(Xk, 1, IJ);
    end
    Gnew_flat = X3.';        % back to I1*I2 x K

    relchange = norm(Gnew_flat-Gflat,'fro') / max(norm(Gflat,'fro'), eps);
    Gflat = Gnew_flat;
    curLoss = loss_fun(Gflat);   % post-projection loss, for next iter's backtracking baseline
    if relchange < reltol
        break;
    end
end

Ghat_flat = Gflat;
end
