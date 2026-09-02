function Ghat = gs_estimator_1D(Y, Bc, S, t0)
% GS_ESTIMATOR_1D  Biased Gerchberg-Saxton channel estimator, adapted from
% Cui et al. "Towards Atomic MIMO Receivers" Algorithm 1 (signal detection)
% to channel estimation by swapping the roles of "unknown symbol vector"
% and "unknown channel row": for each antenna i independently, solve
%   y_i = |S^T g_i + b_i + n_i|
% for the K x 1 channel row g_i, using the KNOWN pilot matrix S in place
% of Cui's known channel A (A = conj(S) under this mapping), and Cui's
% unknown symbol vector s replaced by our unknown channel row g_i.
%
% Inputs:
%   Y  (I x P real)    - observed magnitudes
%   Bc (I x P complex) - true complex reference term b_{i,p} (NOT abs(Bc):
%                         GS operates on the exact model, Eq.(8), not the
%                         linearized one)
%   S  (K x P complex) - pilot matrix
%   t0                 - number of GS iterations (Cui et al. use t0=50)
%
% Output:
%   Ghat (I x K complex) - estimated channel

[I, P] = size(Y);
K = size(S,1);

M    = conj(S)*S.';     % K x K, = A*A^H under A=conj(S); reused for every antenna
Minv = inv(M);
Sc   = conj(S);          % K x P, "A" in Cui's notation

Ghat = zeros(I,K);
for i = 1:I
    yi = Y(i,:).';        % P x 1
    bi = Bc(i,:).';       % P x 1

    % --- spectral initialization (Cui et al., Eq.26 + Algorithm 1 steps 1-4) ---
    % BUGFIX: Cui's Abar=[A^H,b]^H has its first K rows equal to A directly (not
    % conj(A)) -- traced from Abar^H=[A^H,b] (so Abar's first K ROWS, i.e. Abar^H's
    % first K COLUMNS, must equal A^H, giving Abar(1:K,:)=A). Under A=conj(S), that
    % means Sbar's top block must be conj(S), not S -- this was wrong (used bare S).
    Sbar  = [conj(S); conj(bi).'];        % (K+1) x P, augmented pilot matrix
    Mfull = Sbar * diag(yi) * Sbar';      % (K+1) x (K+1), Hermitian PSD
    [V, D] = eig(Mfull);
    [~, idx] = max(real(diag(D)));
    v = V(:, idx);                        % principal eigenvector

    num  = abs(v'*Sbar) * yi;             % scalar
    den  = norm(Sbar'*v)^2;
    rbar = num/den;
    s0   = rbar * v;                      % (K+1) x 1

    g = exp(-1i*angle(s0(K+1))) * s0(1:K);   % de-rotate by the bias-direction phase

    % --- alternating phase / least-squares iteration (Algorithm 1 steps 5-8) ---
    for t = 1:t0
        theta = angle(S.'*g + bi);                 % P x 1
        g = Minv * (Sc * (yi.*exp(1i*theta) - bi)); % K x 1
    end

    Ghat(i,:) = g.';
end

end
