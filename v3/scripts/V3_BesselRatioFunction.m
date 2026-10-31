classdef V3_BesselRatioFunction < deep.DifferentiableFunction
    methods
        function fcn = V3_BesselRatioFunction()
            fcn@deep.DifferentiableFunction(1, SaveInputsForBackward=true, SaveOutputsForBackward=true);
        end

        function ratio = forward(~, kappa)
            ratio = besseli(1, kappa, 1)./besseli(0, kappa, 1);
        end

        function dLdKappa = backward(~, dLdRatio, computeGradients, kappa, ratio)
            if computeGradients(1)
                ratio_derivative = zeros(size(kappa), 'like', kappa);
                small_values = abs(kappa) < 1e-4;
                regular_values = ~small_values;
                ratio_derivative(small_values) = 0.5 - 3*kappa(small_values).^2/16 + 5*kappa(small_values).^4/96;
                ratio_derivative(regular_values) = 1 - ratio(regular_values)./kappa(regular_values) - ratio(regular_values).^2;
                dLdKappa = dLdRatio.*ratio_derivative;
            else
                dLdKappa = [];
            end
        end
    end
end
