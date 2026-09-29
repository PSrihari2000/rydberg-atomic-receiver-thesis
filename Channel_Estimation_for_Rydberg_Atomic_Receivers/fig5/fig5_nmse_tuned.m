%% fig5_nmse_tuned.m
% Fig. 5 with the same settings as the tuned Fig. 4 (fig4/fig4_nmse_tuned.m):
% 50 GD/PGD iterations, RSR = 40 dB, polarization per path.
% These are chosen settings, not values reported in the paper;
% state them in any figure caption.
%
% Runs fig5_nmse.m with the settings below; results are saved to their own
% files (the settings are part of the file name).

clearvars; clc;
RSR_dB = 40;          % reference-to-signal ratio [dB]
POL    = 'path';      % polarization drawn once per path, eq. (374)
T_gd   = 50;          % GD iterations (one value, or one per P in P_list)
T_pgd  = 50;          % PGD iterations (one value, or one per P in P_list)
fig5_nmse
