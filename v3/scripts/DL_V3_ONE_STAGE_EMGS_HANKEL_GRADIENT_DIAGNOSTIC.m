%% Check one EM-GS and Hankel-shrinkage stage end to end
% Uses one deterministic synthetic N=8, K=1, P=10 example.
% It loads no training, validation, or test data and writes no files.
% The custom Bessel-ratio class must be in this script folder.
% Local functions are required for dlfeval and finite-difference calculations.

clear;
clc;
rng(20261028, 'twister');

fprintf('Rydberg V3 one-stage EM-GS/Hankel gradient diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('Synthetic data only; no data files are loaded.\n\n');

N = 8;
K = 1;
P = 10;
p_H = 3;
sigma2 = 0.4;
finite_difference_step = 1e-6;
true_channel = randn(N, K) + 1i*randn(N, K);
initial_channel = true_channel + 0.15*(randn(N, K) + 1i*randn(N, K));
pilots = (randn(K, P) + 1i*randn(K, P))/sqrt(2);
reference_vector = 1.5*(randn(N, 1) + 1i*randn(N, 1))/sqrt(2);
reference = reference_vector*ones(1, P);
noise = sqrt(sigma2/2)*(randn(N, P) + 1i*randn(N, P));
measurements = abs(true_channel*pilots + reference + noise);
threshold_value = 0.3;
number_of_hankel_rows = N-p_H;
number_of_hankel_columns = p_H+1;
hankel_map = zeros(number_of_hankel_rows*number_of_hankel_columns, N);
for row_index = 1:number_of_hankel_rows
    for column_index = 1:number_of_hankel_columns
        linear_index = row_index+(column_index-1)*number_of_hankel_rows;
        antenna_index = row_index+column_index-1;
        hankel_map(linear_index, antenna_index) = 1;
    end
end
hankel_diagonal_counts = sum(hankel_map, 1).';

try
    [loss, gradient_initial_channel, gradient_threshold] = dlfeval(@local_emgs_hankel_loss_and_gradients, dlarray(initial_channel), dlarray(threshold_value), measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, true_channel, p_H, N, K);
    gradient_initial_channel = double(extractdata(gradient_initial_channel));
    gradient_threshold = double(extractdata(gradient_threshold));

    selected_antenna = 4;
    real_perturbation = zeros(N, K);
    real_perturbation(selected_antenna, 1) = 1;
    imaginary_perturbation = 1i*real_perturbation;
    loss_plus_real = local_emgs_hankel_numeric_loss(initial_channel+finite_difference_step*real_perturbation, threshold_value, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    loss_minus_real = local_emgs_hankel_numeric_loss(initial_channel-finite_difference_step*real_perturbation, threshold_value, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    finite_difference_initial_real = (loss_plus_real-loss_minus_real)/(2*finite_difference_step);
    loss_plus_imaginary = local_emgs_hankel_numeric_loss(initial_channel+finite_difference_step*imaginary_perturbation, threshold_value, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    loss_minus_imaginary = local_emgs_hankel_numeric_loss(initial_channel-finite_difference_step*imaginary_perturbation, threshold_value, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    finite_difference_initial_imaginary = (loss_plus_imaginary-loss_minus_imaginary)/(2*finite_difference_step);
    loss_plus_threshold = local_emgs_hankel_numeric_loss(initial_channel, threshold_value+finite_difference_step, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    loss_minus_threshold = local_emgs_hankel_numeric_loss(initial_channel, threshold_value-finite_difference_step, measurements, pilots, reference, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, true_channel);
    finite_difference_threshold = (loss_plus_threshold-loss_minus_threshold)/(2*finite_difference_step);

    autodiff_initial_real = real(gradient_initial_channel(selected_antenna, 1));
    autodiff_initial_imaginary = imag(gradient_initial_channel(selected_antenna, 1));
    autodiff_threshold_real = real(gradient_threshold);
    error_initial_real = abs(autodiff_initial_real-finite_difference_initial_real)/max([1 abs(autodiff_initial_real) abs(finite_difference_initial_real)]);
    error_initial_imaginary = abs(autodiff_initial_imaginary-finite_difference_initial_imaginary)/max([1 abs(autodiff_initial_imaginary) abs(finite_difference_initial_imaginary)]);
    error_threshold = abs(autodiff_threshold_real-finite_difference_threshold)/max([1 abs(autodiff_threshold_real) abs(finite_difference_threshold)]);

    fprintf('One-stage relative channel loss       = %.12g\n', extractdata(loss));
    fprintf('Initial channel real grad / FD         = %.12g / %.12g\n', autodiff_initial_real, finite_difference_initial_real);
    fprintf('Initial channel imaginary grad / FD    = %.12g / %.12g\n', autodiff_initial_imaginary, finite_difference_initial_imaginary);
    fprintf('Threshold real grad / FD              = %.12g / %.12g\n', autodiff_threshold_real, finite_difference_threshold);
    fprintf('Relative errors [G real, G imag, tau] = [%.6g %.6g %.6g]\n', error_initial_real, error_initial_imaginary, error_threshold);
    if all(isfinite([error_initial_real error_initial_imaginary error_threshold])) && max([error_initial_real error_initial_imaginary error_threshold]) < 1e-3
        fprintf('RESULT: PASS. One EM-GS/Hankel stage gradients match finite differences.\n');
    else
        fprintf('RESULT: REVIEW. At least one end-to-end gradient failed the 1e-3 check.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. The one-stage EM-GS/Hankel gradient path raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [loss, gradient_G0, gradient_threshold] = local_emgs_hankel_loss_and_gradients(G0, threshold, Y, S, B, sigma2, hankel_map, hankel_diagonal_counts, Gtrue, p_H, N, K)
right_inverse = S'/(S*S');
field = G0*S+B;
kappa = 2/max(sigma2, eps)*Y.*abs(field);
bessel_ratio_operation = V3_BesselRatioFunction();
bessel_ratio = bessel_ratio_operation(kappa);
G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
G_output = zeros(N, K, 'like', G_emgs);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_emgs(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    shrink_factors = max(1-threshold./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
loss = sum(abs(G_output-Gtrue).^2, 'all')/sum(abs(Gtrue).^2, 'all');
gradient_G0 = dlgradient(loss, G0, 'RetainData', true);
gradient_threshold = dlgradient(loss, threshold);
end

function loss = local_emgs_hankel_numeric_loss(G0, threshold, Y, S, B, sigma2, hankel_map, hankel_diagonal_counts, p_H, N, K, Gtrue)
right_inverse = S'/(S*S');
field = G0*S+B;
kappa = 2/max(sigma2, eps)*Y.*abs(field);
bessel_ratio = besseli(1, kappa, 1)./max(besseli(0, kappa, 1), realmin);
G_emgs = (Y.*exp(1i*angle(field)).*bessel_ratio-B)*right_inverse;
G_output = zeros(N, K);
for user_index = 1:K
    hankel_matrix = reshape(hankel_map*G_emgs(:, user_index), N-p_H, p_H+1);
    gram_matrix = hankel_matrix'*hankel_matrix;
    [eigenvectors, eigenvalue_matrix] = eig(gram_matrix);
    eigenvalues = real(sum(eigenvalue_matrix, 2));
    singular_values = sqrt(max(eigenvalues, 0));
    safe_singular_values = max(singular_values, 1e-12);
    shrink_factors = max(1-threshold./safe_singular_values, 0);
    scaled_eigenvectors = eigenvectors.*reshape(shrink_factors, 1, []);
    hankel_shrunk = hankel_matrix*scaled_eigenvectors*eigenvectors';
    G_output(:, user_index) = (hankel_map'*hankel_shrunk(:))./hankel_diagonal_counts;
end
loss = sum(abs(G_output-Gtrue).^2, 'all')/sum(abs(Gtrue).^2, 'all');
end
