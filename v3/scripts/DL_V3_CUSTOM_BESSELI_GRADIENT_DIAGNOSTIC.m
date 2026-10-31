%% Check custom backward differentiation for the EM-GS Bessel ratio
% Forward values use MATLAB besseli with scaled Bessel functions.
% The backward rule uses the analytic derivative of I1(kappa)/I0(kappa).
% Uses one synthetic positive scalar only; no datasets are loaded.
% Local function is required because dlgradient runs inside dlfeval.

clear;
clc;

fprintf('Rydberg V3 custom Bessel-ratio gradient diagnostic\n');
fprintf('MATLAB release: %s\n', version('-release'));
fprintf('No data files are loaded.\n\n');

kappa_value = 0.7;
finite_difference_step = 1e-6;
kappa_dl = dlarray(kappa_value);

try
    [ratio_dl, gradient_dl] = dlfeval(@local_custom_bessel_ratio_gradient, kappa_dl);
    ratio_value = double(extractdata(ratio_dl));
    gradient_value = double(extractdata(gradient_dl));
    ratio_plus = besseli(1, kappa_value + finite_difference_step, 1)/besseli(0, kappa_value + finite_difference_step, 1);
    ratio_minus = besseli(1, kappa_value - finite_difference_step, 1)/besseli(0, kappa_value - finite_difference_step, 1);
    finite_difference_gradient = (ratio_plus - ratio_minus)/(2*finite_difference_step);
    absolute_error = abs(gradient_value - finite_difference_gradient);
    relative_error = absolute_error/max([1 abs(gradient_value) abs(finite_difference_gradient)]);

    fprintf('kappa                              = %.12g\n', kappa_value);
    fprintf('Bessel ratio                      = %.12g\n', ratio_value);
    fprintf('Custom backward / finite diff     = %.12g / %.12g\n', gradient_value, finite_difference_gradient);
    fprintf('Gradient absolute error            = %.12g\n', absolute_error);
    fprintf('Gradient relative error            = %.6g\n', relative_error);
    if isfinite(ratio_value) && isfinite(gradient_value) && relative_error < 1e-3
        fprintf('RESULT: PASS. The custom Bessel-ratio gradient matches finite differences.\n');
    else
        fprintf('RESULT: REVIEW. The custom gradient did not meet the finite-difference check.\n');
    end
catch diagnostic_error
    fprintf('RESULT: FAIL. The custom differentiable operation raised an error.\n');
    fprintf('%s\n', diagnostic_error.message);
end

function [ratio, gradient] = local_custom_bessel_ratio_gradient(kappa)
bessel_ratio_operation = V3_BesselRatioFunction();
ratio = bessel_ratio_operation(kappa);
gradient = dlgradient(ratio, kappa);
end
