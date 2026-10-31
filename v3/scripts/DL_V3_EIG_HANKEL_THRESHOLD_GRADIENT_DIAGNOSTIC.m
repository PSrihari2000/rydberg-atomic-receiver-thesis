%% Check gradient of learnable threshold in eig-based Hankel shrinkage
% Uses one synthetic 3-by-3 complex Hankel matrix only.
% Does not load data, train a model, or write result files.
% Local functions are required for dlgradient and dlfeval.

clear;
clc;

fprintf('Rydberg V3 eig-based Hankel threshold-gradient diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('No data files are loaded.\n\n');

sequence = [1+0.2i, 0.3+0.7i, -0.4+0.1i, 0.9-0.5i, -0.2-0.8i];
H0 = [sequence(1) sequence(2) sequence(3); sequence(2) sequence(3) sequence(4); sequence(3) sequence(4) sequence(5)];
target = [0.8+0.1i -0.2+0.3i 0.1-0.4i; 0.3-0.2i 0.5+0.1i -0.1+0.2i; -0.2+0.4i 0.2+0.1i 0.6-0.3i];
threshold_value = 0.4;
finite_difference_step = 1e-6;
H_dl = dlarray(H0);
threshold_dl = dlarray(threshold_value);

try
    [loss, threshold_gradient] = dlfeval(@local_eig_hankel_threshold_loss, H_dl, target, threshold_dl);
    autodiff_threshold_complex = double(extractdata(threshold_gradient));
    autodiff_threshold_real = real(autodiff_threshold_complex);
    autodiff_threshold_imaginary = imag(autodiff_threshold_complex);
    loss_plus = local_eig_hankel_threshold_numeric_loss(H0, target, threshold_value + finite_difference_step);
    loss_minus = local_eig_hankel_threshold_numeric_loss(H0, target, threshold_value - finite_difference_step);
    finite_difference_threshold = double((loss_plus - loss_minus)/(2*finite_difference_step));
    absolute_error = abs(autodiff_threshold_real - finite_difference_threshold);
    error_scale = max(1, max(abs([autodiff_threshold_real finite_difference_threshold])));
    relative_error = absolute_error/error_scale;

    fprintf('Loss at original matrix              = %.12g\n', extractdata(loss));
    fprintf('Threshold autodiff real / finite diff = %.12g / %.12g\n', autodiff_threshold_real, finite_difference_threshold);
    fprintf('Threshold autodiff imaginary part    = %.12g\n', autodiff_threshold_imaginary);
    fprintf('Real-component absolute error         = %.12g\n', absolute_error);
    fprintf('Real-component relative error         = %.6g\n', relative_error);
    if isfinite(autodiff_threshold_real) && isfinite(finite_difference_threshold) && abs(autodiff_threshold_real) > 0 && relative_error < 1e-3
        fprintf('RESULT: PASS. The threshold gradient agrees with central finite differences.\n');
    else
        fprintf('RESULT: REVIEW. The threshold gradient is zero, nonfinite, or fails the 1e-3 relative-error check.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. The threshold-gradient comparison raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [loss, threshold_gradient] = local_eig_hankel_threshold_loss(H, target, threshold)
gram = H' * H;
[eigenvectors, eigenvalue_matrix] = eig(gram);
eigenvalues = real(sum(eigenvalue_matrix, 2));
singular_values = sqrt(max(eigenvalues, 0));
safe_singular_values = max(singular_values, 1e-12);
shrink_factors = max(1 - threshold ./ safe_singular_values, 0);
scaled_eigenvectors = eigenvectors .* reshape(shrink_factors, 1, []);
H_shrunk = H * scaled_eigenvectors * eigenvectors';
loss = sum(abs(H_shrunk - target).^2, 'all');
threshold_gradient = dlgradient(loss, threshold);
end

function loss = local_eig_hankel_threshold_numeric_loss(H, target, threshold)
gram = H' * H;
[eigenvectors, eigenvalue_matrix] = eig(gram);
eigenvalues = real(sum(eigenvalue_matrix, 2));
singular_values = sqrt(max(eigenvalues, 0));
safe_singular_values = max(singular_values, 1e-12);
shrink_factors = max(1 - threshold ./ safe_singular_values, 0);
scaled_eigenvectors = eigenvectors .* reshape(shrink_factors, 1, []);
H_shrunk = H * scaled_eigenvectors * eigenvectors';
loss = sum(abs(H_shrunk - target).^2, 'all');
end
