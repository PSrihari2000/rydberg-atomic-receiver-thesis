"""High-SNR fine-tuning from the selected V8 P=10 checkpoints.

Keeps the 40-layer V8 network and dataset fixed. Fine-tuning gives samples
at 25 and 30 dB four times the per-sample NMSE weight; all other SNRs keep
weight 1. Checkpoint selection uses the worst validation DL-minus-GS gap
across the full SNR grid. Test data is evaluated only after selection.
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
        raise FileNotFoundError(f'Upload {filename} beside this V9 script.')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v8 = import_sibling('unfolded_v8', 'colab_unfolded_dl_v8_continue_p10.py')
v7 = v8.v7
v6 = v8.v6
base = v8.base

DATA_ROOT = v8.DATA_ROOT
SOURCE_ROOT = v8.OUTPUT_ROOT
GS_ROOT = v8.GS_ROOT
OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v9_highsnr_finetune_p10_results'
P = 10
SEEDS = v8.SEEDS
MAX_ADDITIONAL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
INITIAL_LEARNING_RATE = 1e-5
LR_SCHEDULER_PATIENCE = 6
MIN_LEARNING_RATE = 5e-7
BATCH_SIZE = v8.BATCH_SIZE
SNR_GRID = tuple(base.SNR_GRID)
HIGH_SNR_WEIGHT = 4.0
HIGH_SNR_POINTS = (25.0, 30.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def high_snr_weighted_loss(predicted_real, predicted_imag, target_real,
                           target_imag, snr):
    error_power = ((predicted_real - target_real).square()
                   + (predicted_imag - target_imag).square())
    target_power = target_real.square() + target_imag.square()
    per_sample_nmse = (
        error_power.sum(dim=(1, 2))
        / target_power.sum(dim=(1, 2)).clamp_min(1e-12)
    )
    weights = torch.ones_like(snr)
    for point in HIGH_SNR_POINTS:
        weights = torch.where(snr == point, HIGH_SNR_WEIGHT, weights)
    return (weights * per_sample_nmse).sum() / weights.sum().clamp_min(1e-12)


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
             channel_scale, device):
    set_seed(seed + 9000)
    generator = torch.Generator().manual_seed(seed + 9000)
    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=device.type == 'cuda', generator=generator,
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )

    source_file = os.path.join(SOURCE_ROOT, f'model_P10_seed{seed}_v8.pt')
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f'Missing selected V8 model: {source_file}')
    source = torch.load(source_file, map_location='cpu', weights_only=False)
    if not np.isclose(float(source['channel_scale']), channel_scale,
                      rtol=1e-6, atol=0.0):
        raise ValueError('V8 checkpoint channel scale does not match training data.')

    model = base.UnfoldedMagnitudeEstimator(40).to(device)
    model.load_state_dict(source['model_state_dict'], strict=True)
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
        'note': 'Selected V8 checkpoint before V9 fine-tuning.',
    }]
    no_improvement = 0
    resume_path = os.path.join(OUTPUT_ROOT, f'resume_P10_seed{seed}.pt')
    final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v9.pt')
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
        print(f'Resuming seed {seed} at V9 fine-tuning epoch {start_epoch}.')

    for epoch in range(start_epoch, MAX_ADDITIONAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted_real, predicted_imag = model(batch)
            loss = high_snr_weighted_loss(
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
        print(f'P=10 seed={seed} V9_epoch={epoch:03d} '
              f'worst validation DL-GS={score:+.4f} dB '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)

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
        }, resume_path)
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping seed {seed} after {epoch} epochs.')
            break

    # Test set is first loaded after validation checkpoint selection.
    model.load_state_dict(best_state)
    test_source = base.load_split(DATA_ROOT, P, 'test')
    test_data = base.prepare_split(test_source, channel_scale)
    test_loader = DataLoader(
        base.ChannelDataset(test_data), batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=device.type == 'cuda',
    )
    test_metrics = base.evaluate(model, test_loader, device)
    gs_test = load_gs_curve('test')
    test_gaps = {
        str(snr): float(test_metrics['nmse_db_by_snr'][str(snr)]
                        - gs_test['nmse_db_by_snr'][str(snr)])
        for snr in SNR_GRID
    }
    result = {
        'P': P, 'seed': seed, 'source_model': source_file,
        'best_v9_additional_epoch': best_epoch,
        'best_worst_validation_gap_to_gs_db': best_score,
        'selected_validation_dl_minus_gs_db_by_snr': best_gaps,
        'test_metrics': test_metrics,
        'test_dl_minus_gs_db_by_snr': test_gaps,
        'loss_weights': {'25_dB': HIGH_SNR_WEIGHT, '30_dB': HIGH_SNR_WEIGHT,
                         'all_other_snr': 1.0},
        'history': history,
    }
    torch.save({
        'model_state_dict': model.state_dict(), 'channel_scale': channel_scale,
        'number_of_layers': 40, 'result': result,
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
    validation_source = base.load_split(DATA_ROOT, P, 'validation')
    validation_data = base.prepare_split(validation_source, channel_scale)
    gs_validation = load_gs_curve('validation')

    summary_path = os.path.join(OUTPUT_ROOT, 'summary_unfolded_v9_highsnr_p10.json')
    results = []
    for seed in SEEDS:
        final_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_v9.pt')
        if os.path.isfile(final_path):
            saved = torch.load(final_path, map_location='cpu', weights_only=False)
            results.append(saved['result'])
            print(f'Using completed V9 result for seed {seed}.')
        else:
            started = time.time()
            result = run_seed(seed, train_data, validation_data,
                              gs_validation, channel_scale, device)
            result['seconds'] = time.time() - started
            results.append(result)
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print(f"Seed {seed} test DL-GS gaps:",
              results[-1]['test_dl_minus_gs_db_by_snr'])
    print(f'Saved V9 high-SNR fine-tuning results to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
