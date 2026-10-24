function [G_hat, selected_rank] = hs_gs_v9_sample(Y, S, B, sigma2, P, sample_index, p_H, cadzow_sweeps, em_updates, rank_selection_updates, training_fraction, seed_base)
% Estimate one channel. Select its Hankel rank from held-out pilot magnitudes.
% The true channel is never used inside this function.
N = size(Y, 1);
rank_list = 1:min(N - p_H, p_H + 1);
if cadzow_sweeps == 0
    selected_rank = rank_list(1);
else
    rng(seed_base + 1000 * P + sample_index, 'twister');
    pilot_order = randperm(P);
    training_count = round(training_fraction * P);
    training_pilots = pilot_order(1:training_count);
    held_out_pilots = pilot_order(training_count + 1:P);
    rank_prediction_error = zeros(length(rank_list), 1);
    for rank_index = 1:length(rank_list)
        G_candidate = run_hs_gs(Y(:, training_pilots), S(:, training_pilots), B(:, training_pilots), sigma2, rank_list(rank_index), rank_selection_updates, cadzow_sweeps, p_H);
        predicted_magnitude = abs(G_candidate * S(:, held_out_pilots) + B(:, held_out_pilots));
        rank_prediction_error(rank_index) = norm(Y(:, held_out_pilots) - predicted_magnitude, 'fro')^2;
    end
    [~, best_rank_index] = min(rank_prediction_error);
    selected_rank = rank_list(best_rank_index);
end
G_hat = run_hs_gs(Y, S, B, sigma2, selected_rank, em_updates, cadzow_sweeps, p_H);
end

function G_hat = run_hs_gs(Y, S, B, sigma2, rank_value, em_updates, cadzow_sweeps, p_H)
N = size(Y, 1);
K = size(S, 1);
G_hat = zeros(N, K);
for cell_index = 1:N
    y_i = Y(cell_index, :).';
    b_i = B(cell_index, :).';
    A = [S.', b_i];
    M = A' * diag(y_i) * A;
    [V, D] = eig(M);
    [~, principal_index] = max(real(diag(D)));
    v = V(:, principal_index);
    v = v / max(norm(v), eps);
    r = (abs(A * v).' * y_i) / max(norm(A * v)^2, eps);
    G_hat(cell_index, :) = (r * exp(-1j * angle(v(K + 1))) * v(1:K)).';
end
for update_index = 1:em_updates
    field = G_hat * S + B;
    kappa = (2 / max(sigma2, eps)) * Y .* abs(field);
    ratio = besseli(1, kappa, 1) ./ max(besseli(0, kappa, 1), realmin);
    expected_field = Y .* exp(1j * angle(field)) .* ratio;
    G_tilde = (expected_field - B) * S' / (S * S');
    G_processed = zeros(N, K);
    for user_index = 1:K
        channel_vector = G_tilde(:, user_index);
        for sweep_index = 1:cadzow_sweeps
            H_rows = N - p_H;
            H_columns = p_H + 1;
            H = zeros(H_rows, H_columns);
            for row_index = 1:H_rows
                H(row_index, :) = channel_vector(row_index:row_index + H_columns - 1).';
            end
            [left_vectors, singular_values, right_vectors] = svd(H, 'econ');
            H_rank = left_vectors(:, 1:rank_value) * singular_values(1:rank_value, 1:rank_value) * right_vectors(:, 1:rank_value)';
            reconstructed_channel = zeros(N, 1);
            copy_count = zeros(N, 1);
            for row_index = 1:H_rows
                for column_index = 1:H_columns
                    sample_index = row_index + column_index - 1;
                    reconstructed_channel(sample_index) = reconstructed_channel(sample_index) + H_rank(row_index, column_index);
                    copy_count(sample_index) = copy_count(sample_index) + 1;
                end
            end
            channel_vector = reconstructed_channel ./ max(copy_count, eps);
        end
        G_processed(:, user_index) = channel_vector;
    end
    G_hat = G_processed;
end
end
