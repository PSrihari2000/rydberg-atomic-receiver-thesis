%% Check dlarray gradient support for the EM-GS Bessel ratio
% Uses one positive synthetic scalar only.
% Does not load data, train a model, or change any estimator code.
% A local function is required because dlgradient runs inside dlfeval.

clear;
clc;

fprintf('Rydberg V3 besseli gradient diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('No data files are loaded.\n\n');

kappa_value = 0.7;
finite_difference_step = 1e-6;
kappa_dl = dlarray(kappa_value);

try
    [ratio_dl, gradient_dl] = dlfeval(@local_bessel_ratio_gradient, kappa_dl);
    ratio_value = double(extractdata(ratio_dl));
    gradient_value = double(extractdata(gradient_dl));
    ratio_plus = besseli(1, kappa_value + finite_difference_step, 1)/besseli(0, kappa_value + finite_difference_step, 1);
    ratio_minus = besseli(1, kappa_value - finite_difference_step, 1)/besseli(0, kappa_value - finite_difference_step, 1);
    finite_difference_gradient = (ratio_plus - ratio_minus)/(2*finite_difference_step);
    gradient_real = real(gradient_value);
    absolute_error = abs(gradient_real - finite_difference_gradient);
    relative_error = absolute_error/max([1 abs(gradient_real) abs(finite_difference_gradient)]);

    fprintf('kappa                              = %.12g\n', kappa_value);
    fprintf('Bessel ratio                      = %.12g\n', ratio_value);
    fprintf('Autodiff gradient real / finite diff = %.12g / %.12g\n', gradient_real, finite_difference_gradient);
    fprintf('Autodiff gradient imaginary part  = %.12g\n', imag(gradient_value));
    fprintf('Gradient relative error            = %.6g\n', relative_error);
    if isfinite(ratio_value) && isfinite(gradient_real) && relative_error < 1e-3
        fprintf('RESULT: PASS. dlarray differentiated the Bessel ratio and matched finite differences.\n');
    else
        fprintf('RESULT: REVIEW. The Bessel gradient did not match the finite-difference check.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. Automatic differentiation through besseli raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [ratio, gradient] = local_bessel_ratio_gradient(kappa)
ratio = besseli(1, kappa, 1)./besseli(0, kappa, 1);
gradient = dlgradient(ratio, kappa);
end
