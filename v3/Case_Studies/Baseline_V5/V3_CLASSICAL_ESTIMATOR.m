function [G, iteration_count, converged] = V3_CLASSICAL_ESTIMATOR(method, Y, S, B, sigma2, p_H, shrink_multiplier, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed)
% Shared one-dimensional estimators for validation policy selection and frozen test scoring.
% Inputs are one magnitude-only observation and its known pilots/reference.
% The HS-GS option is soft Hankel-shrinkage EM-GS; multiplier zero is EM-GS.
% GD uses the Xu et al. 1D linearized-gradient update. Its initialization,
% step-size, and convergence settings below are explicit local choices.

method = upper(string(method));
if method == "GS"
    [G, iteration_count, converged] = v3_classical_gs(Y, S, B, maximum_updates, convergence_tolerance);
elseif method == "GD"
    [G, iteration_count, converged] = v3_classical_gd(Y, S, B, maximum_updates, convergence_tolerance, gd_initial_variance, random_seed);
elseif method == "EM-GS"
    [G, iteration_count, converged] = v3_classical_emgs(Y, S, B, sigma2, 0, 0, maximum_updates, convergence_tolerance);
elseif method == "HS-GS"
    [G, iteration_count, converged] = v3_classical_emgs(Y, S, B, sigma2, p_H, shrink_multiplier, maximum_updates, convergence_tolerance);
else
    error('Unknown estimator method: %s', method);
end
end

function [G, iteration_count, converged] = v3_classical_gs(Y, S, B, maximum_updates, convergence_tolerance)
G = v3_classical_spectral_initialize(Y, S, B);
right_inverse = S'/(S*S');
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    previous_G = G;
    G = (Y.*exp(1i*angle(G*S+B))-B)*right_inverse;
    if norm(G-previous_G, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function [G, iteration_count, converged] = v3_classical_gd(Y, S, B, maximum_updates, convergence_tolerance, initial_variance, random_seed)
rng(mod(double(random_seed)+720000, 2^32-1), 'twister');
N = size(Y, 1);
K = size(S, 1);
G = sqrt(initial_variance/2)*(randn(N, K)+1i*randn(N, K));
phase_reference = exp(-1i*angle(B));
step_size = 1/max(real(eig(S*S')));
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    previous_G = G;
    G = G+step_size*((Y-abs(B)-real(phase_reference.*(G*S))).*conj(phase_reference))*S';
    if norm(G-previous_G, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function [G, iteration_count, converged] = v3_classical_emgs(Y, S, B, sigma2, p_H, shrink_multiplier, maximum_updates, convergence_tolerance)
G = v3_classical_spectral_initialize(Y, S, B);
right_inverse = S'/(S*S');
if shrink_multiplier > 0
    threshold = shrink_multiplier*v3_classical_hankel_noise_scale(S, sigma2, size(Y, 1), p_H);
else
    threshold = 0;
end
converged = false;
iteration_count = maximum_updates;
for update_index = 1:maximum_updates
    previous_G = G;
    field = G*S+B;
    kappa = 2/max(sigma2, eps)*Y.*abs(field);
    bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
    G = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
    if threshold > 0
        G = v3_classical_soft_hankel_projection(G, p_H, threshold);
    end
    if norm(G-previous_G, 'fro')^2/max(norm(G, 'fro')^2, eps) < convergence_tolerance
        iteration_count = update_index;
        converged = true;
        break;
    end
end
end

function scale = v3_classical_hankel_noise_scale(S, sigma2, N, p_H)
K = size(S, 1);
right_inverse = S'/(S*S');
scale = sqrt(sigma2)*norm(right_inverse, 'fro')*sqrt(max(N-p_H, p_H+1)/K);
end

function G = v3_classical_soft_hankel_projection(G, p_H, threshold)
N = size(G, 1);
number_of_rows = N-p_H;
number_of_columns = p_H+1;
for user_index = 1:size(G, 2)
    channel_vector = G(:, user_index);
    hankel_matrix = zeros(number_of_rows, number_of_columns);
    for row_index = 1:number_of_rows
        hankel_matrix(row_index, :) = channel_vector(row_index:row_index+number_of_columns-1).';
    end
    [U, Sigma, V] = svd(hankel_matrix, 'econ');
    shrunk_singular_values = max(diag(Sigma)-threshold, 0);
    hankel_matrix = U*diag(shrunk_singular_values)*V';
    reconstructed_channel = zeros(N, 1);
    diagonal_counts = zeros(N, 1);
    for row_index = 1:number_of_rows
        for column_index = 1:number_of_columns
            antenna_index = row_index+column_index-1;
            reconstructed_channel(antenna_index) = reconstructed_channel(antenna_index)+hankel_matrix(row_index, column_index);
            diagonal_counts(antenna_index) = diagonal_counts(antenna_index)+1;
        end
    end
    G(:, user_index) = reconstructed_channel./diagonal_counts;
end
end

function G = v3_classical_spectral_initialize(Y, S, B)
N = size(Y, 1);
K = size(S, 1);
G = zeros(N, K);
for antenna_index = 1:N
    y = Y(antenna_index, :).';
    b = B(antenna_index, :).';
    A = [S.', b];
    M = A'*diag(y)*A;
    [eigenvectors, eigenvalues] = eig(M);
    [~, principal_index] = max(real(diag(eigenvalues)));
    v = eigenvectors(:, principal_index);
    v = v/max(norm(v), eps);
    scale = (abs(A*v).'*y)/max(norm(A*v)^2, eps);
    G(antenna_index, :) = (scale*exp(-1i*angle(v(K+1)))*v(1:K)).';
end
end
