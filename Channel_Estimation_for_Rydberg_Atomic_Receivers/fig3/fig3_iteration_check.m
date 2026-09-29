%% fig3_iteration_check.m
% Which setting gives "floor at P = 10, on the CRLB at P = 30"?
% Compares changing the reference strength (RSR) with changing the number
% of GD / GS iterations. 1D model of fig3_nmse.m with polarization per path,
% exact magnitude measurements y = |GS + B + N| (94).
%
% Output: NMSE in dB of GD and GS at SNR = 10, 20, 30 dB for P = 10 and 30.
% Takes a few minutes (100 trials per point).

clearvars; clc; rng(3);

fprintf('\n(A) RSR sweep, GD 3000 iterations (converged), GS t0 = 50\n');
for RSR = [0 10 20 30]
    pr(sprintf('RSR=%2d dB', RSR), run(RSR, 3000, 50));
end

fprintf('\n(B) iteration budget of both methods, RSR = 30 dB\n');
for T = [[10 10]; [20 20]; [30 30]; [50 50]].'
    pr(sprintf('T_gd=%d T_gs=%d', T(1), T(2)), run(30, T(1), T(2)));
end

fprintf('\n(C) GD converged, only GS iterations changed, RSR = 30 dB\n');
for Tgs = [10 20 30]
    pr(sprintf('T_gd=3000 T_gs=%d', Tgs), run(30, 3000, Tgs));
end

%% ------------------------------------------------------------------------
function pr(lbl, r)
fprintf(['%-18s P=10 GD/GS @10,20,30 dB: %6.1f %6.1f | %6.1f %6.1f | %6.1f %6.1f' ...
         '   P=30: %6.1f %6.1f | %6.1f %6.1f | %6.1f %6.1f\n'], lbl, r(1,:), r(2,:));
end

function r = run(RSR, Tgd, Tgs)
K = 3; I = 8; MC = 100; SNRs = [10 20 30];
q = 1.602e-19; a0 = 5.292e-11; hbar = 1.054571817e-34; mu = [0; 1785.9*q*a0; 0];
r = zeros(2, 6); Ps = [10 30];
for ip = 1:2
    P = Ps(ip); e = zeros(2, 3); en = 0;
    for mc = 1:MC
        [G, S, B] = gen_trial(K, I, P, RSR, mu, hbar);
        A = G*S; Z = exp(-1j*angle(B)); en = en + norm(G, 'fro')^2;
        for is = 1:3
            s2 = mean(abs(A(:)).^2)/10^(SNRs(is)/10);                 % (90)
            Y  = abs(A + B + sqrt(s2/2)*(randn(I, P) + 1j*randn(I, P)));   % (94)
            e(1, is) = e(1, is) + norm(est_gd(Y, S, B, Z, Tgd) - G, 'fro')^2;
            e(2, is) = e(2, is) + norm(est_gs(Y, S, B, Tgs) - G, 'fro')^2;
        end
    end
    e = 10*log10(e/en); r(ip, :) = reshape(e, 1, []);
end
end

function G = est_gd(Y, S, B, Z, T)
% GD on the linearised model, (220), (223), step (418a), fixed T iterations
[I, ~] = size(Y); K = size(S, 1);
G = zeros(I, K); eta = 1/max(eig(S*S')); Yc = Y - abs(B);
for t = 1:T
    E = Yc - real(Z.*(G*S)); G = G + eta*(E.*conj(Z))*S';
end
end

function G = est_gs(Y, S, B, T)
% biased GS: spectral init (351)-(358), then T iterations of (360)-(364)
[I, ~] = size(Y); K = size(S, 1); G = zeros(I, K);
for i = 1:I
    y = Y(i, :).'; Ab = [S.', B(i, :).']; M = Ab'*diag(y)*Ab;
    [V, D] = eig((M + M')/2); [~, m] = max(real(diag(D))); v = V(:, m);
    qq = abs(Ab*v); rr = (qq.'*y)/(qq.'*qq); g = exp(-1j*angle(rr*v(end)))*rr*v;
    G(i, :) = g(1:K).';
end
Si = S'/(S*S');
for t = 1:T
    X = G*S + B; R = Y.*exp(1j*angle(X)); G = (R - B)*Si;
end
end

function [G, S, B] = gen_trial(K, I, P, RSR, mu, hbar)
% channel (75) with polarization per path, pilots (145), reference (88)-(89)
idx = (0:I-1).'; G = zeros(I, K);
for k = 1:K
    L = randi([3 7]); ph = pi*cos(pi*rand(1, L));
    al = (randn(1, L) + 1j*randn(1, L))/sqrt(2); ep = sqrt(1/3)*randn(3, L);
    G(:, k) = exp(-1j*idx*ph)*((mu.'*ep).*al/hbar).';
end
S = (randn(K, P) + 1j*randn(K, P))/sqrt(2);
phb = pi*cos(pi*rand); ab = sqrt(10)*(randn + 1j*randn)/sqrt(2); eb = sqrt(1/3)*randn(3, 1);
B = (mu.'*eb/hbar)*ab*exp(-1j*idx*phb)*ones(1, P); A = G*S;
B = B*sqrt(10^(RSR/10)*mean(abs(A(:)).^2)/mean(abs(B(:)).^2));
end
