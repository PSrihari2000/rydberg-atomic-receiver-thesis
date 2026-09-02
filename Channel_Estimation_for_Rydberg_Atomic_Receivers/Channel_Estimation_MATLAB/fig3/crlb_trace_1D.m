function crb_total = crlb_trace_1D(sigma2, S, I)
% CRLB_TRACE_1D  trace(CRB(g)) for the 1D array, Eq.(31) with I1*I2 -> I.
% CRB(g) = 4*sigma2*(I_I kron (S*S^T)^-1), block-diagonal with I identical
% K x K blocks, so trace(CRB(g)) = I * 4*sigma2*trace((conj(S)*S.')^-1).
%
% IMPORTANT: sigma2 here must be the REAL-valued residual noise variance that
% Eq.28-31 are derived for (i.e. the variance of n_bar in Eq.11), NOT the
% variance of the original complex noise n~CN(0,sigma2) from Eq.8 -- those
% two differ by a factor of 2 (Eq.11: n_bar ~ N(0,sigma2/2)). Callers working
% from the complex-model sigma2 must pass sigma2/2 in here.

K = size(S,1); %#ok<NASGU>
M = conj(S)*S.';           % K x K, = S* S^T
crb_total = I * 4 * sigma2 * real(trace(inv(M)));

end
