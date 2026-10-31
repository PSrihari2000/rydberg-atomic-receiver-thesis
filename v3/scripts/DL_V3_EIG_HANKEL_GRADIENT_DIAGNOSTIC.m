%% R2026b eig-based Hankel shrinkage gradient check
% This diagnostic uses only a small synthetic 3-by-3 complex matrix.
% It does not load training, validation, or test data.
% It does not train a network or change any project results.
% A local function is required because MATLAB requires dlgradient to run
% inside a function evaluated by dlfeval.

clear;
clc;

fprintf('Rydberg V3 eig-based Hankel gradient diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('No data files are loaded.\n\n');

sequence = [1+0.2i, 0.3+0.7i, -0.4+0.1i, 0.9-0.5i, -0.2-0.8i];
H0 = [sequence(1) sequence(2) sequence(3); sequence(2) sequence(3) sequence(4); sequence(3) sequence(4) sequence(5)];
target = [0.8+0.1i -0.2+0.3i 0.1-0.4i; 0.3-0.2i 0.5+0.1i -0.1+0.2i; -0.2+0.4i 0.2+0.1i 0.6-0.3i];
threshold = 0.4;
H_dl = dlarray(H0);

try
    [loss, gradient] = dlfeval(@local_eig_hankel_loss, H_dl, target, threshold);
    loss_value = extractdata(loss);
    gradient_value = extractdata(gradient);
    gradient_is_finite = all(isfinite(gradient_value), 'all');
    gradient_norm = norm(gradient_value, 'fro');
    fprintf('Real scalar loss                 = %.12g\n', loss_value);
    fprintf('Gradient finite                  = %d\n', gradient_is_finite);
    fprintf('Gradient Frobenius norm          = %.12g\n', gradient_norm);
    if isfinite(loss_value) && gradient_is_finite && gradient_norm > 0
        fprintf('RESULT: PASS. MATLAB produced a finite nonzero gradient through eig-based Hankel shrinkage.\n');
    else
        fprintf('RESULT: FAIL. The loss or gradient is not finite and nonzero.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. The eig-based gradient path raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [loss, gradient] = local_eig_hankel_loss(H, target, threshold)
gram = H' * H;
[eigenvectors, eigenvalue_matrix] = eig(gram);
eigenvalues = real(sum(eigenvalue_matrix, 2));
singular_values = sqrt(max(eigenvalues, 0));
safe_singular_values = max(singular_values, 1e-12);
shrink_factors = max(1 - threshold ./ safe_singular_values, 0);
scaled_eigenvectors = eigenvectors .* reshape(shrink_factors, 1, []);
H_shrunk = H * scaled_eigenvectors * eigenvectors';
loss = sum(abs(H_shrunk - target).^2, 'all');
gradient = dlgradient(loss, H, 'AllowComplex', true);
end
