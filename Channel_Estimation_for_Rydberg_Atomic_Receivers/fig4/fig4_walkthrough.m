%% fig4_walkthrough.m
% Step-by-step walkthrough of the 2D model and one PGD run on a tiny array,
% so that every matrix fits on the screen. Every printed quantity is
% labelled with the equation number of ce_derivation_1D_2D_PGD_CRLB.pdf.
%
% With STEP = true the script pauses after each step; press any key.
% To run without pauses:  STEP = false; fig4_walkthrough

clearvars -except STEP; clc;
if ~exist('STEP', 'var'), STEP = true; end
rng(1);
format short g

%% Step 0: parameters
I1 = 3; I2 = 3;                 % 3 x 3 array, Na = 9 cells
K  = 2;                         % users
Lk = [1 2];                     % paths: user 1 has 1 path, user 2 has 2 paths
P  = 6;                         % pilots (P >= 2K = 4)
SNR_dB = 20;  RSR_dB = 40;
d1 = 0.5; d2 = 0.5;             % spacing / lambda
q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34;
mu_eg = [0; 1785.9*q*a0; 0];
Na = I1*I2;  i1 = (0:I1-1).';  i2 = (0:I2-1).';

fprintf('==================== STEP 0: parameters ====================\n');
fprintf('array %d x %d (Na = %d), K = %d users, L_k = %s, P = %d\n', I1, I2, Na, K, mat2str(Lk), P);
fprintf('SNR = %g dB, RSR = %g dB; polarization drawn once per path, eq. (374)\n', SNR_dB, RSR_dB);
stepPause(STEP);

%% Step 1: each user's I1 x I2 channel, eqs. (370)-(382)
Gk = cell(1, K);
fprintf('==================== STEP 1: channel slices G_k ====================\n');
for k = 1:K
    Gk{k} = zeros(I1, I2);
    for l = 1:Lk(k)
        theta = pi*rand;  phi = 2*pi*rand;
        u = 2*pi*d1*cos(theta);  v = 2*pi*d2*sin(theta)*cos(phi);       % (370)
        a1 = exp(-1j*i1*u);  a2 = exp(-1j*i2*v);                        % (376), (377)
        alpha = (randn + 1j*randn)/sqrt(2);
        beta  = (mu_eg.'*(sqrt(1/3)*randn(3, 1)))*alpha/hbar;           % (373)
        Gkl = beta*(a1*a2.');                                           % (378)
        Gk{k} = Gk{k} + Gkl;                                            % (380)
        fprintf('\nuser %d, path %d: theta = %.1f deg, phi = %.1f deg, u = %.4f, v = %.4f\n', ...
            k, l, rad2deg(theta), rad2deg(phi), u, v);
        show('(376) a1', a1);  show('(377) a2', a2);
        show(sprintf('(378) G_%d^(%d) = beta a1 a2^T', k, l), Gkl);
        fprintf('(379) rank(G_%d^(%d)) = %d\n', k, l, rank(Gkl));
    end
    show(sprintf('(380) G_%d = sum of its paths', k), Gk{k});
    fprintf('singular values of G_%d: %s\n', k, mat2str(svd(Gk{k}).', 4));
    fprintf('(382) rank(G_%d) = %d  (L_%d = %d; the rest are ~0 up to rounding)\n', ...
        k, rank(Gk{k}), k, Lk(k));
end
stepPause(STEP);

%% Step 2: mode-3 unfolding, eqs. (393)-(403)
G3 = zeros(K, Na);
for k = 1:K, G3(k, :) = reshape(Gk{k}, 1, []); end         % (398a), column-major
fprintf('==================== STEP 2: unfolding ====================\n');
show('(398a) G_(3)  (K x I1I2), row k = vec(G_k)^T', G3);
fprintf('column index q = i1 + (i2-1) I1, eq. (398): e.g. (i1,i2) = (2,3) -> q = %d\n', 2 + (3-1)*I1);
for k = 1:K
    fprintf('(402) mat(row %d) equals G_%d?  max difference = %.1e\n', ...
        k, k, max(abs(reshape(G3(k,:), I1, I2) - Gk{k}), [], 'all'));
end
stepPause(STEP);

%% Step 3: pilots, reference, noise, measurements, eqs. (383)-(401)
S  = (randn(K, P) + 1j*randn(K, P))/sqrt(2);                % (395)
A3 = S.'*G3;                                                % (400), P x Na
th_b = pi*rand; ph_b = 2*pi*rand;
u_b = 2*pi*d1*cos(th_b); v_b = 2*pi*d2*sin(th_b)*cos(ph_b);
g_b = reshape((mu_eg.'*(sqrt(1/3)*randn(3, 1)))/hbar*sqrt(10)*(randn + 1j*randn)/sqrt(2) ...
      *exp(-1j*i1*u_b)*exp(-1j*i2*v_b).', 1, []);           % (383), one LoS path
B3 = ones(P, 1)*g_b;                                        % s_b,p = 1
B3 = B3*sqrt(10^(RSR_dB/10)*mean(abs(A3(:)).^2)/mean(abs(B3(:)).^2));
Z3 = exp(-1j*angle(B3));                                    % (384), (399)
sigma2 = mean(abs(A3(:)).^2)/10^(SNR_dB/10);                % (390)
N3 = sqrt(sigma2/2)*(randn(P, Na) + 1j*randn(P, Na));
Y3 = abs(A3 + B3 + N3);                                     % (385), exact magnitude
Ylin = real(Z3.*A3) + abs(B3) + real(Z3.*N3);               % (401), linearised

fprintf('==================== STEP 3: measurements ====================\n');
show('(395) S  (K x P)', S);
show('(400) S^T G_(3)  (P x I1I2)', A3);
show('(399) |B_(3)|', abs(B3));
show('(385) Y_(3) = |S^T G_(3) + B_(3) + N_(3)|', Y3);
fprintf('rms(Y_exact - Y_lin) / sigma_r = %.3f  (small -> linearisation (401) accurate)\n', ...
    sqrt(mean((Y3(:) - Ylin(:)).^2))/sqrt(sigma2/2));
stepPause(STEP);

%% Step 4: one PGD iteration in full, eqs. (415)-(428)
zeta = 1/max(eig(S*S'));                                    % (418a)
G = zeros(K, Na);                                           % start from zero
E    = Y3 - abs(B3) - real(Z3.*(S.'*G));                    % (415)
grad = -conj(S)*(E.*conj(Z3));                              % (416)
M    = G - zeta*grad;                                       % (417)-(418)

fprintf('==================== STEP 4: one PGD iteration ====================\n');
fprintf('(418a) step size zeta = 1/lambda_max(S S^H) = %.4g\n', zeta);
show('(415) residual E  (P x I1I2, real)', E);
show('(416) gradient  (K x I1I2)', grad);
show('(418) M = G - zeta*gradient', M);
for k = 1:K
    Mk = reshape(M(k, :), I1, I2);                          % (423)
    [U, Sg, V] = svd(Mk);                                   % (424)
    L  = Lk(k);
    Mk_L = U(:,1:L)*Sg(1:L,1:L)*V(:,1:L)';                  % (426)
    fprintf('\n--- user %d (L_%d = %d) ---\n', k, k, L);
    show(sprintf('(423) M_%d = mat(row %d of M)', k, k), Mk);
    fprintf('(425) singular values of M_%d: %s   -> rank %d before projection\n', ...
        k, mat2str(diag(Sg).', 4), rank(Mk));
    show('(424) U', U); show('(424) V', V);
    fprintf('check (424): max |U S V^H - M_%d| = %.1e\n', k, max(abs(U*Sg*V' - Mk), [], 'all'));
    fprintf('check (424): max |U S V^T - M_%d| = %.1e  (V^T instead of V^H is wrong)\n', ...
        k, max(abs(U*Sg*V.' - Mk), [], 'all'));
    show(sprintf('(426) M_%d^(L) = U(:,1:L) S(1:L,1:L) V(:,1:L)^H', k), Mk_L);
    fprintf('rank after projection = %d  (should be %d)\n', rank(Mk_L), L);
    fprintf('(427) ||M^(L) - M||_F^2 = %.6e,  sum of discarded sigma^2 = %.6e\n', ...
        norm(Mk_L - Mk, 'fro')^2, sum(diag(Sg(L+1:end, L+1:end)).^2));
    M(k, :) = reshape(Mk_L, 1, []);                         % (428)
end
stepPause(STEP);

%% Step 5: PGD and GD iterations, eq. (429a)
fprintf('==================== STEP 5: iterations ====================\n');
fprintf('   t    cost PGD      NMSE PGD[dB]  rank(G_1,G_2) |  cost GD       NMSE GD[dB]   rank(G_1,G_2)\n');
Gp = zeros(K, Na); Gg = zeros(K, Na); Yc = Y3 - abs(B3);
for t = 1:200
    for m = 1:2                                            % m = 1 PGD, m = 2 GD
        if m == 1, G = Gp; else, G = Gg; end
        E = Yc - real(Z3.*(S.'*G));
        M = G + zeta*conj(S)*(E.*conj(Z3));
        if m == 1
            for k = 1:K
                [U, Sg, V] = svd(reshape(M(k, :), I1, I2)); L = Lk(k);
                M(k, :) = reshape(U(:,1:L)*Sg(1:L,1:L)*V(:,1:L)', 1, []);
            end
            Gp = M;
        else
            Gg = M;
        end
    end
    if any(t == [1 2 3 5 10 20 50 100 200])
        cp = norm(Yc - real(Z3.*(S.'*Gp)), 'fro')^2;  cg = norm(Yc - real(Z3.*(S.'*Gg)), 'fro')^2;
        np = 10*log10(norm(Gp - G3, 'fro')^2/norm(G3, 'fro')^2);
        ng = 10*log10(norm(Gg - G3, 'fro')^2/norm(G3, 'fro')^2);
        rp = [rank(reshape(Gp(1,:), I1, I2)), rank(reshape(Gp(2,:), I1, I2))];
        rg = [rank(reshape(Gg(1,:), I1, I2)), rank(reshape(Gg(2,:), I1, I2))];
        fprintf('%4d   %10.3e   %9.2f      %s        |  %10.3e   %9.2f     %s\n', ...
            t, cp, np, mat2str(rp), cg, ng, mat2str(rg));
    end
end
fprintf('\nPGD keeps rank(G_k) = L_k = %s at every iteration; GD ends up full rank.\n', mat2str(Lk));
fprintf('CRLB (454),(457a): NMSE = %.2f dB\n', ...
    10*log10(2*sigma2*Na*real(trace(inv(conj(S)*S.')))/norm(G3, 'fro')^2));

%% Local functions
function show(label, X)
fprintf('\n%s  [%d x %d]\n', label, size(X,1), size(X,2));
disp(X);
end

function stepPause(STEP)
if STEP
    fprintf('\n--- press any key for the next step ---\n');
    pause;
end
end
