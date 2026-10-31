%% Compare eig-based Hankel shrinkage gradient with finite differences
% Uses only one synthetic 3-by-3 complex Hankel matrix.
% Does not load data, train a model, or write result files.
% Local functions are required for the dlfeval and numeric loss calculations.

clear;
clc;

fprintf('Rydberg V3 eig-based Hankel finite-difference diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('No data files are loaded.\n\n');

sequence = [1+0.2i, 0.3+0.7i, -0.4+0.1i, 0.9-0.5i, -0.2-0.8i];
H0 = [sequence(1) sequence(2) sequence(3); sequence(2) sequence(3) sequence(4); sequence(3) sequence(4) sequence(5)];
target = [0.8+0.1i -0.2+0.3i 0.1-0.4i; 0.3-0.2i 0.5+0.1i -0.1+0.2i; -0.2+0.4i 0.2+0.1i 0.6-0.3i];
threshold = 0.4;
finite_difference_step = 1e-6;
H_dl = dlarray(H0);

try
    [loss, gradient] = dlfeval(@local_eig_hankel_loss_and_gradient, H_dl, target, threshold);
    gradient_value = extractdata(gradient);
    selected_row = 2;
    selected_column = 2;
    real_direction = zeros(size(H0));
    real_direction(selected_row, selected_column) = 1;
    imaginary_direction = zeros(size(H0));
    imaginary_direction(selected_row, selected_column) = 1i;

    loss_plus_real = local_eig_hankel_numeric_loss(H0 + finite_difference_step*real_direction, target, threshold);
    loss_minus_real = local_eig_hankel_numeric_loss(H0 - finite_difference_step*real_direction, target, threshold);
    finite_difference_real = (loss_plus_real - loss_minus_real)/(2*finite_difference_step);
    loss_plus_imaginary = local_eig_hankel_numeric_loss(H0 + finite_difference_step*imaginary_direction, target, threshold);
    loss_minus_imaginary = local_eig_hankel_numeric_loss(H0 - finite_difference_step*imaginary_direction, target, threshold);
    finite_difference_imaginary = (loss_plus_imaginary - loss_minus_imaginary)/(2*finite_difference_step);

    autodiff_real = real(gradient_value(selected_row, selected_column));
    autodiff_imaginary = imag(gradient_value(selected_row, selected_column));
    real_relative_error = abs(autodiff_real - finite_difference_real)/max([1 abs(autodiff_real) abs(finite_difference_real)]);
    imaginary_relative_error = abs(autodiff_imaginary - finite_difference_imaginary)/max([1 abs(autodiff_imaginary) abs(finite_difference_imaginary)]);

    fprintf('Loss at original matrix              = %.12g\n', extractdata(loss));
    fprintf('Perturbed matrix entry                = H(%d,%d)\n', selected_row, selected_column);
    fprintf('Real part: autodiff / finite diff     = %.12g / %.12g\n', autodiff_real, finite_difference_real);
    fprintf('Real part relative error              = %.6g\n', real_relative_error);
    fprintf('Imag part: autodiff / finite diff     = %.12g / %.12g\n', autodiff_imaginary, finite_difference_imaginary);
    fprintf('Imag part relative error              = %.6g\n', imaginary_relative_error);
    if real_relative_error < 1e-3 && imaginary_relative_error < 1e-3
        fprintf('RESULT: PASS. Both gradient components agree with central finite differences.\n');
    else
        fprintf('RESULT: REVIEW. The gradient does not meet the 1e-3 relative-error check.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. The gradient comparison raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [loss, gradient] = local_eig_hankel_loss_and_gradient(H, target, threshold)
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

function loss = local_eig_hankel_numeric_loss(H, target, threshold)
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
