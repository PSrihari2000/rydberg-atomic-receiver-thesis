"""Controlled V10 continuation for P=10 (validation only).

Continues each selected V10 checkpoint for up to 100 more epochs using a
smaller learning rate. The architecture, training set, high-SNR weighted loss,
and worst-SNR validation checkpoint criterion stay fixed. This experiment is
designed to determine whether extra optimization alone closes the high-SNR
gap. The test split is never loaded.

Upload this script beside the V10, V9, V8, V7, V6, V5, and V3 scripts in
Colab. The V10 script reads the datasets/checkpoints from their existing Drive
paths.
"""

import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def import_sibling(module_name, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Upload {filename} beside this V11 script.')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v10 = import_sibling(
    'unfolded_v10', 'colab_unfolded_dl_v10_pilot_conditioned_p10.py'
)
v9 = v10.v9
base = v10.base

DATA_ROOT = v10.DATA_ROOT
SOURCE_ROOT = v10.OUTPUT_ROOT
GS_ROOT = v10.GS_ROOT
OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v11_continue_v10_p10_results'
P = 10
SEEDS = v10.SEEDS
SNR_GRID = tuple(v10.SNR_GRID)
BATCH_SIZE = v10.BATCH_SIZE
MAX_ADDITIONAL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
INITIAL_LEARNING_RATE = 1e-6
LR_SCHEDULER_PATIENCE = 8
MIN_LEARNING_RATE = 1e-7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def run_seed(seed, train_data, validation_data, gs_validation,
             channel_scale, condition_mean, condition_std, device):
    set_seed(seed + 11000)
    generator = torch.Generator().manual_seed(seed + 11000)
    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=device.type == 'cuda', generator=generator,
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )

    source_path = os.path.join(SOURCE_ROOT, f'model_P10_seed{seed}_v10.pt')
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f'Missing V10 selected checkpoint: {source_path}. '
            'Complete the V10 run and make its results folder visible in Drive.'
        )
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if not np.isclose(float(source['channel_scale']), channel_scale,
                      rtol=1e-6, atol=0.0):
        raise ValueError(f'V10 checkpoint channel scale mismatch for seed {seed}.')
    if not np.isclose(float(source['condition_mean']), condition_mean,
                      rtol=1e-6, atol=1e-8):
        raise ValueError(f'V10 condition mean mismatch for seed {seed}.')
    if not np.isclose(float(source['condition_std']), condition_std,
                      rtol=1e-6, atol=1e-8):
        raise ValueError(f'V10 condition scale mismatch for seed {seed}.')

    model = v10.PilotConditionedUnfoldedEstimator(40).to(device)
    model.load_state_dict(source['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    resume_path = os.path.join(OUTPUT_ROOT, f'resume_P10_seed{seed}_v11.pt')
    final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v11.pt')
    summary_path = os.path.join(
        OUTPUT_ROOT, 'summary_unfolded_v11_continue_v10_p10.json'
    )

    _, best_gaps, best_score = v10.validation_gaps(
        model, validation_loader, gs_validation, device
    )
    best_state = cpu_state_dict(model)
    best_epoch = 0
    history = [{
        'additional_epoch': 0,
        'validation_dl_minus_gs_db_by_snr': best_gaps,
        'worst_validation_gap_to_gs_db': best_score,
        'note': 'Selected V10 checkpoint before continuation.',
    }]
    no_improvement = 0
    start_epoch = 1

    if os.path.isfile(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        best_score = float(saved['best_score'])
        best_state = saved['best_state']
        best_epoch = int(saved['best_epoch'])
        best_gaps = saved['best_gaps']
        history = saved['history']
        no_improvement = int(saved['no_improvement'])
        start_epoch = int(saved['epoch']) + 1
        if not np.isclose(saved['condition_mean'], condition_mean,
                          rtol=1e-6, atol=1e-8):
            raise ValueError('Resume checkpoint condition mean mismatch.')
        if not np.isclose(saved['condition_std'], condition_std,
                          rtol=1e-6, atol=1e-8):
            raise ValueError('Resume checkpoint condition scale mismatch.')
        print(f'Resuming seed {seed} at continuation epoch {start_epoch}.')

    started = time.time()
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
            running_loss += float(loss.item())

        validation, gaps, score = v10.validation_gaps(
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
            f'P=10 seed={seed} V11_epoch={epoch:03d} '
            f'worst validation DL-GS={score:+.5f} dB '
            f'gap25={gaps["25"]:+.5f} dB gap30={gaps["30"]:+.5f} dB '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True,
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_gaps = gaps
            no_improvement = 0
        else:
            no_improvement += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
            'best_state': best_state,
            'best_epoch': best_epoch,
            'best_gaps': best_gaps,
            'no_improvement': no_improvement,
            'history': history,
            'condition_mean': condition_mean,
            'condition_std': condition_std,
        }, resume_path)

        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping seed {seed} at continuation epoch {epoch}.')
            break

    model.load_state_dict(best_state, strict=True)
    result = {
        'P': P,
        'seed': seed,
        'source_v10_checkpoint': source_path,
        'additional_epochs_run': len(history) - 1,
        'best_continuation_epoch': best_epoch,
        'best_worst_validation_gap_to_gs_db': best_score,
        'selected_validation_dl_minus_gs_db_by_snr': best_gaps,
        'learning_rate': INITIAL_LEARNING_RATE,
        'architecture_changed': False,
        'test_evaluated': False,
        'seconds': time.time() - started,
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
    raw_condition = v10.raw_log_pilot_condition(train_source)
    condition_mean = float(np.mean(raw_condition))
    condition_std = max(float(np.std(raw_condition)), 1e-6)
    train_data['pilot_condition'] = (
        (raw_condition - condition_mean) / condition_std
    ).astype(np.float32)

    validation_source = base.load_split(DATA_ROOT, P, 'validation')
    validation_data = base.prepare_split(validation_source, channel_scale)
    v10.add_condition_feature(
        validation_data, validation_source, condition_mean, condition_std
    )
    gs_validation = v10.load_gs_curve('validation')

    summary_path = os.path.join(
        OUTPUT_ROOT, 'summary_unfolded_v11_continue_v10_p10.json'
    )
    results = []
    for seed in SEEDS:
        final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v11.pt')
        if os.path.isfile(final_path):
            saved = torch.load(final_path, map_location='cpu', weights_only=False)
            results.append(saved['result'])
            print(f'Using completed V11 result for seed {seed}.')
        else:
            result = run_seed(
                seed, train_data, validation_data, gs_validation,
                channel_scale, condition_mean, condition_std, device,
            )
            results.append(result)
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print(f'Seed {seed} selected validation gaps:',
              results[-1]['selected_validation_dl_minus_gs_db_by_snr'])

    print(f'Saved validation-only V11 continuation results to {OUTPUT_ROOT}')
    print('The test split was not loaded or evaluated.')


if __name__ == '__main__':
    main()
