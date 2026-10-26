"""SNR-aware pure unfolded channel-estimation ablations for Colab.

Run three controlled models for each pilot count P=10 and P=30:
  1. learned initializer, fixed learned update parameters;
  2. zero initializer, update parameters conditioned on measured SNR;
  3. learned initializer and SNR-conditioned update parameters.

The magnitude measurement equation remains inside the unfolded iterations.
There is no GS initialization, Hankel projection, or classical output.
Checkpoint selection minimizes the worst (largest) per-SNR validation NMSE
in dB. The validation macro-average is recorded as a secondary metric.
Independent test data is evaluated only after checkpoint selection.

Upload this script beside colab_unfolded_dl_v3_highsnr.py in Colab. The v3
script supplies the shared dataset reader, preprocessing, and evaluation.
"""

import copy
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


BASE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'colab_unfolded_dl_v3_highsnr.py',
)
if not os.path.isfile(BASE_SCRIPT):
    raise FileNotFoundError(
        'Upload colab_unfolded_dl_v3_highsnr.py beside this v5 script.'
    )

spec = importlib.util.spec_from_file_location('unfolded_dl_v3_base', BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v5_snr_aware_results'
base.NUMBER_OF_LAYERS = 40
base.NUMBER_OF_PARAMETERS_PER_LAYER = 8
base.BATCH_SIZE = 128
base.NUMBER_OF_EPOCHS = 100
base.LEARNING_RATE = 3e-4
base.PATIENCE = 15
base.SEED = 20261025

ABLATIONS = [
    {'name': 'initializer_only', 'learned_initializer': True,
     'snr_conditioned_updates': False},
    {'name': 'snr_conditioned_only', 'learned_initializer': False,
     'snr_conditioned_updates': True},
    {'name': 'initializer_plus_snr_conditioning', 'learned_initializer': True,
     'snr_conditioned_updates': True},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SNRConditionedUnfoldedEstimator(nn.Module):
    """Learned initialization plus differentiable magnitude-consistency steps."""

    def __init__(self, P, N, K, number_of_layers,
                 learned_initializer, snr_conditioned_updates):
        super().__init__()
        self.N = N
        self.K = K
        self.learned_initializer = learned_initializer
        self.snr_conditioned_updates = snr_conditioned_updates

        # Eight base values per layer, plus eight SNR slopes when enabled.
        initial = torch.cat([
            torch.zeros(number_of_layers, 1),
            -torch.ones(number_of_layers, 1),
            -3.0 * torch.ones(number_of_layers, 1),
            torch.zeros(number_of_layers, 5),
        ], dim=1)
        self.update_parameters = nn.Parameter(initial)
        if snr_conditioned_updates:
            self.snr_slopes = nn.Parameter(torch.zeros_like(initial))
        else:
            self.register_parameter('snr_slopes', None)

        if learned_initializer:
            feature_count = 3 * N * P + 2 * K * P + 1
            self.initializer = nn.Sequential(
                nn.LayerNorm(feature_count),
                nn.Linear(feature_count, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 2 * N * K),
            )
            # Start close to the old zero initialization while allowing the
            # initializer to learn a useful basin from the first updates.
            nn.init.normal_(self.initializer[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.initializer[-1].bias)
        else:
            self.initializer = None

    def forward(self, batch):
        y = batch['y']
        s_real = batch['s_real']
        s_imag = batch['s_imag']
        b_real = batch['b_real']
        b_imag = batch['b_imag']
        noise_ratio = batch['noise_ratio'].view(-1, 1, 1)
        pilot_bound = batch['pilot_bound'].view(-1, 1, 1).clamp_min(1e-6)
        batch_size = y.shape[0]

        # SNR is known from the controlled simulation; this feature is scaled
        # to approximately [-1, 1] over the experiment's -5 to 30 dB grid.
        snr_feature = ((batch['snr'].view(-1, 1) - 12.5) / 17.5).clamp(-2.0, 2.0)

        if self.learned_initializer:
            features = torch.cat([
                y.flatten(start_dim=1),
                s_real.flatten(start_dim=1),
                s_imag.flatten(start_dim=1),
                b_real.flatten(start_dim=1),
                b_imag.flatten(start_dim=1),
                snr_feature,
            ], dim=1)
            initial_channel = self.initializer(features).view(
                batch_size, self.N, 2 * self.K
            )
            real_channel = initial_channel[:, :, :self.K]
            imag_channel = initial_channel[:, :, self.K:]
        else:
            real_channel = torch.zeros(
                batch_size, self.N, self.K, device=y.device, dtype=y.dtype
            )
            imag_channel = torch.zeros_like(real_channel)

        previous_real_channel = real_channel
        previous_imag_channel = imag_channel

        for layer_index in range(self.update_parameters.shape[0]):
            parameters = self.update_parameters[layer_index]
            if self.snr_conditioned_updates:
                parameters = parameters + self.snr_slopes[layer_index] * snr_feature

            step_size = 1.9 * torch.sigmoid(parameters[..., 0])
            momentum = 0.95 * torch.sigmoid(parameters[..., 1])
            ridge_weight = torch.nn.functional.softplus(parameters[..., 2])
            data_gain = 2.0 * torch.sigmoid(parameters[..., 5])
            gradient_gain = 2.0 * torch.sigmoid(parameters[..., 6])
            update_gain = 2.0 * torch.sigmoid(parameters[..., 7])

            # Scalar or per-example parameter values broadcast over N and K.
            step_size = step_size.reshape(-1, 1, 1) if step_size.ndim else step_size
            momentum = momentum.reshape(-1, 1, 1) if momentum.ndim else momentum
            ridge_weight = ridge_weight.reshape(-1, 1, 1) if ridge_weight.ndim else ridge_weight
            data_gain = data_gain.reshape(-1, 1, 1) if data_gain.ndim else data_gain
            gradient_gain = gradient_gain.reshape(-1, 1, 1) if gradient_gain.ndim else gradient_gain
            update_gain = update_gain.reshape(-1, 1, 1) if update_gain.ndim else update_gain

            intermediate_real = real_channel + momentum * (
                real_channel - previous_real_channel
            )
            intermediate_imag = imag_channel + momentum * (
                imag_channel - previous_imag_channel
            )
            field_real = (
                torch.bmm(intermediate_real, s_real)
                - torch.bmm(intermediate_imag, s_imag) + b_real
            )
            field_imag = (
                torch.bmm(intermediate_real, s_imag)
                + torch.bmm(intermediate_imag, s_real) + b_imag
            )
            amplitude = torch.sqrt(
                field_real.square() + field_imag.square() + 1e-12
            )
            residual_factor = data_gain * (amplitude - y) / amplitude.clamp_min(1e-6)
            error_real = residual_factor * field_real
            error_imag = residual_factor * field_imag
            s_real_t = s_real.transpose(1, 2)
            s_imag_t = s_imag.transpose(1, 2)
            gradient_real = (
                torch.bmm(error_real, s_real_t)
                + torch.bmm(error_imag, s_imag_t)
            )
            gradient_imag = (
                -torch.bmm(error_real, s_imag_t)
                + torch.bmm(error_imag, s_real_t)
            )

            previous_real_channel = real_channel
            previous_imag_channel = imag_channel
            denominator = 1.0 + step_size * ridge_weight * noise_ratio / pilot_bound
            real_channel = (
                intermediate_real
                - update_gain * step_size * gradient_gain * gradient_real / pilot_bound
            ) / denominator
            imag_channel = (
                intermediate_imag
                - update_gain * step_size * gradient_gain * gradient_imag / pilot_bound
            ) / denominator

        return real_channel, imag_channel


def train_one_model(P, ablation, device):
    # Reset seed so ablations start from reproducible, comparable conditions.
    set_seed(base.SEED + P)
    train_source = base.load_split(base.DATA_ROOT, P, 'train')
    validation_source = base.load_split(base.DATA_ROOT, P, 'validation')
    test_source = base.load_split(base.DATA_ROOT, P, 'test')
    channel_scale = float(np.sqrt(np.mean(np.abs(train_source['G_data']) ** 2)))
    channel_scale = max(channel_scale, np.sqrt(np.finfo(np.float32).eps))
    train_data = base.prepare_split(train_source, channel_scale)
    validation_data = base.prepare_split(validation_source, channel_scale)
    test_data = base.prepare_split(test_source, channel_scale)

    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=base.BATCH_SIZE,
        shuffle=True, pin_memory=True,
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=base.BATCH_SIZE,
        shuffle=False, pin_memory=True,
    )
    test_loader = DataLoader(
        base.ChannelDataset(test_data), batch_size=base.BATCH_SIZE,
        shuffle=False, pin_memory=True,
    )

    N = train_data['y'].shape[1]
    K = train_data['s_real'].shape[1]
    model = SNRConditionedUnfoldedEstimator(
        P, N, K, base.NUMBER_OF_LAYERS,
        ablation['learned_initializer'], ablation['snr_conditioned_updates'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE)
    model_tag = ablation['name']
    checkpoint_file = os.path.join(
        base.OUTPUT_ROOT, f'checkpoint_P{P}_{model_tag}.pt'
    )
    best_worst_snr = np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, base.NUMBER_OF_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_real, pred_imag = model(batch)
            loss = base.weighted_nmse_loss(
                pred_real, pred_imag,
                batch['target_real'], batch['target_imag'], batch['snr'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        validation = base.evaluate(model, validation_loader, device)
        per_snr = validation['nmse_db_by_snr']
        missing = [str(snr) for snr in base.SNR_GRID if str(snr) not in per_snr]
        if missing:
            raise ValueError(f'Validation set lacks SNR groups: {missing}')
        macro_snr = float(np.mean([per_snr[str(snr)] for snr in base.SNR_GRID]))
        worst_snr = float(max(per_snr.values()))
        record = {
            'epoch': epoch,
            'train_loss': running_loss / max(len(train_loader), 1),
            'validation_macro_mean_nmse_db': macro_snr,
            'validation_worst_snr_nmse_db': worst_snr,
            'validation_nmse_db_by_snr': per_snr,
        }
        history.append(record)
        print(
            f"P={P} {model_tag} epoch={epoch:03d} "
            f"macro={macro_snr:.3f} dB worst={worst_snr:.3f} dB"
        )

        if worst_snr < best_worst_snr:
            best_worst_snr = worst_snr
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_worst_snr': best_worst_snr,
            'best_state': best_state,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
        }, checkpoint_file)
        if epochs_without_improvement >= base.PATIENCE:
            print(f'Early stopping P={P} {model_tag} at epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError(f'No validation checkpoint selected for P={P}, {model_tag}.')
    model.load_state_dict(best_state)
    # Test split is used only once, after validation checkpoint selection.
    test = base.evaluate(model, test_loader, device)
    output_file = os.path.join(
        base.OUTPUT_ROOT, f'model_P{P}_{model_tag}.pt'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'channel_scale': channel_scale,
        'P': P,
        'N': N,
        'K': K,
        'number_of_layers': base.NUMBER_OF_LAYERS,
        'ablation': ablation,
        'selection_metric': 'minimum worst-SNR validation NMSE in dB',
        'best_worst_snr_validation_nmse_db': best_worst_snr,
        'history': history,
        'test_metrics': test,
    }, output_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    return {
        'P': P,
        'ablation': ablation,
        'best_worst_snr_validation_nmse_db': best_worst_snr,
        'history': history,
        'test_metrics': test,
        'model_file': output_file,
    }


def main():
    os.makedirs(base.OUTPUT_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('Pure unfolded estimator; no GS initialization or Hankel projection.')
    results = []
    for P in base.P_LIST:
        for ablation in ABLATIONS:
            started = time.time()
            result = train_one_model(P, ablation, device)
            result['seconds'] = time.time() - started
            results.append(result)
            print(
                f"P={P} {ablation['name']} test NMSE="
                f"{result['test_metrics']['nmse_db']:.3f} dB"
            )
            print('Test NMSE by SNR:', result['test_metrics']['nmse_db_by_snr'])

    summary_file = os.path.join(base.OUTPUT_ROOT, 'summary_unfolded_v5_snr_aware.json')
    with open(summary_file, 'w', encoding='utf-8') as file_handle:
        json.dump(results, file_handle, indent=2)
    print(f'Saved v5 ablations to: {base.OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
