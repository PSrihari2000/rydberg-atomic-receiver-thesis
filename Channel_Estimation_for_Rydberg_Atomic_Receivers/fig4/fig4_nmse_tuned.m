%% fig4_nmse_tuned.m
% Fig. 4 with a limited iteration budget, to reproduce the shape of Fig. 4 of
% Xu et al. (error floor at P = 10, CRLB reached at P = 30).
% The iteration counts, the reference strength (RSR) and the per-path
% polarization are chosen settings, not values reported in the paper;
% state them in any figure caption.
%
% Runs fig4_nmse.m with the settings below; results are saved to their own
% files (the settings are part of the file name).

clearvars; clc;
RSR_dB = 40;          % reference-to-signal ratio [dB]
POL    = 'path';      % polarization drawn once per path, eq. (374)
T_gd   = 50;          % GD iterations (one value, or one per P in P_list)
T_pgd  = 50;          % PGD iterations (one value, or one per P in P_list)
fig4_nmse
