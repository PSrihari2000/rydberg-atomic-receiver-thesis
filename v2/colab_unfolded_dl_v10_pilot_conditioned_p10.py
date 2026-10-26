"""V10 P=10 pilot-conditioned fine-tuning, initialized from V9.

Adds a normalized log pilot-condition feature to the six learned update
controls at every unfolded layer. The new coefficients start at exactly zero,
so before training V10 reproduces its matching V9 model. The 40-layer backbone,
training data, V9 loss weights, and validation checkpoint criterion are kept
fixed. This script does not open or evaluate the test split.
"""

import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def import_sibling(module_name, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Upload {filename} beside this V10 script.')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v9 = import_sibling('unfolded_v9', 'colab_unfolded_dl_v9_highsnr_finetune.py')
v8 = v9.v8
v7 = v9.v7
v6 = v9.v6
base = v9.base

DATA_ROOT = v9.DATA_ROOT
SOURCE_ROOT = v9.OUTPUT_ROOT
GS_ROOT = v9.GS_ROOT
OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v10_pilot_conditioned_p10_results'
P = 10
SEEDS = v9.SEEDS
MAX_ADDITIONAL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
INITIAL_LEARNING_RATE = 1e-5
LR_SCHEDULER_PATIENCE = 6
MIN_LEARNING_RATE = 5e-7
BATCH_SIZE = v9.BATCH_SIZE
SNR_GRID = tuple(base.SNR_GRID)


class PilotConditionedUnfoldedEstimator(nn.Module):
    """V9's 40-layer estimator with zero-initialized pilot-condition controls."""

    def __init__(self, number_of_layers=40):
        super().__init__()
        self.number_of_layers = number_of_layers
        self.parameters_per_layer = nn.Parameter(torch.zeros(number_of_layers, 8))
        # Controls step, momentum, ridge, data gain, gradient gain, update gain.
        # At zero, all six controls are identical to the V9 model.
        self.condition_coefficients = nn.Parameter(torch.zeros(number_of_layers, 6))

    def forward(self, batch):
        y = batch['y']
        s_real = batch['s_real']
        s_imag = batch['s_imag']
        b_real = batch['b_real']
        b_imag = batch['b_imag']
        noise_ratio = batch['noise_ratio'].view(-1, 1, 1)
        pilot_bound = batch['pilot_bound'].view(-1, 1, 1).clamp_min(1e-6)
        condition = batch['pilot_condition'].view(-1, 1, 1)

        batch_size, n_antennas, n_pilots = y.shape
        n_paths = s_real.shape[1]
        real_channel = torch.zeros(
            batch_size, n_antennas, n_paths, device=y.device, dtype=y.dtype
        )
        imag_channel = torch.zeros_like(real_channel)
        previous_real_channel = torch.zeros_like(real_channel)
        previous_imag_channel = torch.zeros_like(real_channel)
        noise_level = torch.log1p(noise_ratio / pilot_bound)

        for layer_index in range(self.number_of_layers):
            p = self.parameters_per_layer[layer_index]
            c = self.condition_coefficients[layer_index]
            step_size = 1.9 * torch.sigmoid(
                p[0] + p[3] * noise_level + c[0] * condition
            )
            momentum = 0.95 * torch.sigmoid(
                p[1] + p[4] * noise_level + c[1] * condition
            )
            ridge_weight = F.softplus(p[2] + c[2] * condition)
            data_gain = 2.0 * torch.sigmoid(p[5] + c[3] * condition)
            gradient_gain = 2.0 * torch.sigmoid(p[6] + c[4] * condition)
            update_gain = 2.0 * torch.sigmoid(p[7] + c[5] * condition)

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
            s_real_transpose = s_real.transpose(1, 2)
            s_imag_transpose = s_imag.transpose(1, 2)
            gradient_real = (
                torch.bmm(error_real, s_real_transpose)
                + torch.bmm(error_imag, s_imag_transpose)
            )
            gradient_imag = (
                -torch.bmm(error_real, s_imag_transpose)
                + torch.bmm(error_imag, s_real_transpose)
            )
            previous_real_channel = real_channel
            previous_imag_channel = imag_channel
            normalized_gradient_real = gradient_gain * gradient_real / pilot_bound
            normalized_gradient_imag = gradient_gain * gradient_imag / pilot_bound
            denominator = 1.0 + step_size * ridge_weight * noise_ratio / pilot_bound
            real_channel = (
                intermediate_real - update_gain * step_size * normalized_gradient_real
            ) / denominator
            imag_channel = (
                intermediate_imag - update_gain * step_size * normalized_gradient_imag
            ) / denominator

        return real_channel, imag_channel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_log_pilot_condition(source):
    s = source['S_data'].astype(np.complex128)
    gram = s @ np.swapaxes(s.conj(), 1, 2)
    eigenvalues = np.linalg.eigvalsh(gram).real
    smallest = np.maximum(eigenvalues[:, 0], 1e-12)
    largest = np.maximum(eigenvalues[:, -1], smallest)
    return (0.5 * np.log(largest / smallest)).astype(np.float32)


def add_condition_feature(data, source, condition_mean, condition_std):
    raw = raw_log_pilot_condition(source)
    data['pilot_condition'] = ((raw - condition_mean) / condition_std).astype(np.float32)
    return data


def load_gs_curve(split):
    path = os.path.join(GS_ROOT, f'gs_{split}_P10.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing cached GS curve: {path}')
    with open(path, 'r', encoding='utf-8') as handle:
        curve = json.load(handle)
    dataset_path = os.path.join(DATA_ROOT, f'{split}_P10.mat')
    if v6.sha256_file(dataset_path) != curve.get('dataset_sha256'):
        raise ValueError(f'The cached GS curve does not match {dataset_path}.')
    return curve


def validation_gaps(model, loader, gs_validation, device):
    metrics = base.evaluate(model, loader, device)
    gaps = {
        str(snr): float(metrics['nmse_db_by_snr'][str(snr)]
                        - gs_validation['nmse_db_by_snr'][str(snr)])
        for snr in SNR_GRID
    }
    return metrics, gaps, max(gaps.values())


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def run_seed(seed, train_data, validation_data, gs_validation,
             channel_scale, condition_mean, condition_std, device):
    set_seed(seed + 10000)
    generator = torch.Generator().manual_seed(seed + 10000)
    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=device.type == 'cuda', generator=generator,
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )

    source_file = os.path.join(SOURCE_ROOT, f'model_P10_seed{seed}_v9.pt')
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f'Missing selected V9 model: {source_file}')
    source = torch.load(source_file, map_location='cpu', weights_only=False)
    if not np.isclose(float(source['channel_scale']), channel_scale,
                      rtol=1e-6, atol=0.0):
        raise ValueError('V9 checkpoint channel scale does not match training data.')

    model = PilotConditionedUnfoldedEstimator(40).to(device)
    loaded = model.load_state_dict(source['model_state_dict'], strict=False)
    if loaded.unexpected_keys or loaded.missing_keys != ['condition_coefficients']:
        raise ValueError(
            f'Unexpected V9-to-V10 transfer: missing={loaded.missing_keys}, '
            f'unexpected={loaded.unexpected_keys}'
        )
    with torch.no_grad():
        if not torch.equal(model.condition_coefficients,
                           torch.zeros_like(model.condition_coefficients)):
            raise ValueError('V10 pilot-condition coefficients must start at zero.')

    reference_model = base.UnfoldedMagnitudeEstimator(40).to(device)
    reference_model.load_state_dict(source['model_state_dict'], strict=True)
    reference_model.eval()
    model.eval()
    example_batch = base.move_batch_to_device(next(iter(validation_loader)), device)
    with torch.no_grad():
        reference_prediction = reference_model(example_batch)
        v10_initial_prediction = model(example_batch)
    max_initial_difference = max(
        float(torch.max(torch.abs(a - b)).item())
        for a, b in zip(reference_prediction, v10_initial_prediction)
    )
    # The added zero-valued condition terms can change float32 operation
    # rounding by a few ULPs. The measured difference should remain negligible.
    INITIALIZATION_ATOL = 1e-5
    if max_initial_difference > INITIALIZATION_ATOL:
        raise ValueError(
            'Zero-initialized V10 does not reproduce V9 before training: '
            f'max abs difference={max_initial_difference}'
        )
    print(f'Seed {seed} V10=V9 initialization check: '
          f'max abs difference {max_initial_difference:.3g}')
    del reference_model, example_batch, reference_prediction, v10_initial_prediction

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    _, initial_gaps, best_score = validation_gaps(
        model, validation_loader, gs_validation, device
    )
    best_state = cpu_state_dict(model)
    best_epoch = 0
    best_gaps = initial_gaps
    history = [{
        'additional_epoch': 0,
        'validation_dl_minus_gs_db_by_snr': initial_gaps,
        'worst_validation_gap_to_gs_db': best_score,
        'note': 'V9 initialization; pilot-condition coefficients are zero.',
    }]
    no_improvement = 0
    resume_path = os.path.join(OUTPUT_ROOT, f'resume_P10_seed{seed}.pt')
    final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v10.pt')
    start_epoch = 1

    if os.path.isfile(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        best_score, best_state = saved['best_score'], saved['best_state']
        best_epoch, best_gaps = saved['best_epoch'], saved['best_gaps']
        history, no_improvement = saved['history'], saved['no_improvement']
        start_epoch = saved['epoch'] + 1
        if not np.isclose(saved['condition_mean'], condition_mean, rtol=1e-6, atol=1e-8):
            raise ValueError('Resume checkpoint condition mean differs from current train data.')
        if not np.isclose(saved['condition_std'], condition_std, rtol=1e-6, atol=1e-8):
            raise ValueError('Resume checkpoint condition scale differs from current train data.')
        print(f'Resuming seed {seed} at V10 fine-tuning epoch {start_epoch}.')

    for epoch in range(start_epoch, MAX_ADDITIONAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted_real, predicted_imag = model(batch)
            loss = v9.high_snr_weighted_loss(
                predicted_real, predicted_imag,
                batch['target_real'], batch['target_imag'], batch['snr'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        validation, gaps, score = validation_gaps(
            model, validation_loader, gs_validation, device
        )
        scheduler.step(score)
        history.append({
            'additional_epoch': epoch,
            'train_loss': running_loss / max(len(train_loader), 1),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_nmse_db_by_snr': validation['nmse_db_by_snr'],
            'validation_dl_minus_gs_db_by_snr': gaps,
            'worst_validation_gap_to_gs_db': score,
        })
        print(
            f'P=10 seed={seed} V10_epoch={epoch:03d} '
            f'worst validation DL-GS={score:+.5f} dB '
            f'gap25={gaps["25"]:+.5f} dB gap30={gaps["30"]:+.5f} dB '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True,
        )

        if score < best_score:
            best_score, best_epoch, best_state, best_gaps = (
                score, epoch, cpu_state_dict(model), gaps
            )
            no_improvement = 0
        else:
            no_improvement += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score, 'best_state': best_state,
            'best_epoch': best_epoch, 'best_gaps': best_gaps,
            'no_improvement': no_improvement, 'history': history,
            'condition_mean': condition_mean, 'condition_std': condition_std,
        }, resume_path)
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping seed {seed} after {epoch} V10 epochs.')
            break

    model.load_state_dict(best_state)
    result = {
        'P': P, 'seed': seed, 'source_model': source_file,
        'best_v10_additional_epoch': best_epoch,
        'best_worst_validation_gap_to_gs_db': best_score,
        'selected_validation_dl_minus_gs_db_by_snr': best_gaps,
        'condition_feature': 'standardized log condition number of S, train mean/std',
        'condition_mean': condition_mean, 'condition_std': condition_std,
        'loss_weights': {'25_dB': v9.HIGH_SNR_WEIGHT,
                         '30_dB': v9.HIGH_SNR_WEIGHT,
                         'all_other_snr': 1.0},
        'test_evaluated': False,
        'history': history,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'channel_scale': channel_scale,
        'condition_mean': condition_mean,
        'condition_std': condition_std,
        'number_of_layers': 40,
        'result': result,
    }, final_path)
    return result


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_source = base.load_split(DATA_ROOT, P, 'train')
    channel_scale = float(np.sqrt(np.mean(np.abs(train_source['G_data']) ** 2)))
    channel_scale = max(channel_scale, np.sqrt(np.finfo(np.float32).eps))
    train_data = base.prepare_split(train_source, channel_scale)
    train_condition = raw_log_pilot_condition(train_source)
    condition_mean = float(np.mean(train_condition))
    condition_std = max(float(np.std(train_condition)), 1e-6)
    train_data['pilot_condition'] = (
        (train_condition - condition_mean) / condition_std
    ).astype(np.float32)

    validation_source = base.load_split(DATA_ROOT, P, 'validation')
    validation_data = base.prepare_split(validation_source, channel_scale)
    add_condition_feature(
        validation_data, validation_source, condition_mean, condition_std
    )
    gs_validation = load_gs_curve('validation')

    summary_path = os.path.join(
        OUTPUT_ROOT, 'summary_unfolded_v10_pilot_condition_p10.json'
    )
    results = []
    for seed in SEEDS:
        final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v10.pt')
        if os.path.isfile(final_path):
            saved = torch.load(final_path, map_location='cpu', weights_only=False)
            results.append(saved['result'])
            print(f'Using completed V10 validation result for seed {seed}.')
        else:
            started = time.time()
            result = run_seed(
                seed, train_data, validation_data, gs_validation,
                channel_scale, condition_mean, condition_std, device,
            )
            result['seconds'] = time.time() - started
            results.append(result)
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print(f"Seed {seed} selected validation gaps:",
              results[-1]['selected_validation_dl_minus_gs_db_by_snr'])

    print(f'Saved V10 validation-only results to {OUTPUT_ROOT}')
    print('No test split was opened or evaluated by this script.')


if __name__ == '__main__':
    main()
