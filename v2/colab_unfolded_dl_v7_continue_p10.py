"""Continue fine-tuning all three v6 P=10 models for up to 100 more epochs.

Each run starts from its corresponding selected v6 checkpoint. The learning
rate starts lower than v6 and drops when the worst per-SNR validation gap to
GS plateaus. Checkpoints are selected by minimizing max_SNR(DL_dB - GS_dB).
The test split is evaluated only after each continuation selects its model.

Upload this script beside v3, v5, and v6 scripts in Colab. The v6 P=10 model
files and GS curve caches must already be in their Drive results folder.
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
        raise FileNotFoundError(f'Upload {filename} beside this v7 script.')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = import_sibling('unfolded_v6', 'colab_unfolded_dl_v6_gs_gap.py')
base = v6.base

DATA_ROOT = v6.DATA_ROOT
SOURCE_ROOT = '/content/gdrive/MyDrive/DL_dataset_RSR25dB/v6_starting_models'
OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v7_p10_extended_results'
P = 10
SEEDS = (20261026, 20261027, 20261028)
MAX_ADDITIONAL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 25
INITIAL_LEARNING_RATE = 5e-5
LR_SCHEDULER_PATIENCE = 8
MIN_LEARNING_RATE = 1e-6
BATCH_SIZE = 128
SNR_GRID = tuple(base.SNR_GRID)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_gs_validation():
    cache_path = os.path.join(SOURCE_ROOT, 'gs_validation_P10.json')
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f'Missing GS validation cache: {cache_path}. Run v6 first.'
        )
    with open(cache_path, 'r', encoding='utf-8') as file_handle:
        result = json.load(file_handle)
    dataset_path = os.path.join(DATA_ROOT, 'validation_P10.mat')
    if v6.sha256_file(dataset_path) != result.get('dataset_sha256'):
        raise ValueError('The saved GS curve is for a different validation dataset.')
    return result


def run_seed(seed, train_data, validation_data, gs_validation, channel_scale, device):
    set_seed(seed + 7000)
    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=BATCH_SIZE,
        shuffle=True, pin_memory=device.type == 'cuda',
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )

    source_file = os.path.join(SOURCE_ROOT, f'model_P10_seed{seed}.pt')
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f'Missing v6 starting model: {source_file}')
    source = torch.load(source_file, map_location='cpu', weights_only=False)
    model = base.UnfoldedMagnitudeEstimator(40).to(device)
    model.load_state_dict(source['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    initial_gaps = source['result']['selected_validation_dl_minus_gs_db_by_snr']
    best_score = max(float(initial_gaps[str(snr)]) for snr in SNR_GRID)
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(OUTPUT_ROOT, f'checkpoint_P10_seed{seed}.pt')
    model_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_extended.pt')
    start_epoch = 1

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_score = checkpoint['best_score']
        best_state = checkpoint['best_state']
        best_epoch = checkpoint['best_epoch']
        history = checkpoint['history']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming seed {seed} at additional epoch {start_epoch}.')

    for epoch in range(start_epoch, MAX_ADDITIONAL_EPOCHS + 1):
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
        dl_curve = validation['nmse_db_by_snr']
        gs_curve = gs_validation['nmse_db_by_snr']
        gaps = {
            str(snr): float(dl_curve[str(snr)] - gs_curve[str(snr)])
            for snr in SNR_GRID
        }
        score = max(gaps.values())
        scheduler.step(score)
        record = {
            'additional_epoch': epoch,
            'source_v6_epoch': source['result']['best_epoch'],
            'train_loss': running_loss / max(len(train_loader), 1),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_nmse_db_by_snr': dl_curve,
            'validation_dl_minus_gs_db_by_snr': gaps,
            'worst_validation_gap_to_gs_db': score,
        }
        history.append(record)
        print(
            f"P=10 seed={seed} extra_epoch={epoch:03d} "
            f"worst DL-GS gap={score:.4f} dB "
            f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True,
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
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
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
            'best_state': best_state,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
        }, checkpoint_path)
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping seed {seed} after {epoch} extra epochs.')
            break

    model.load_state_dict(best_state)
    test_source = base.load_split(DATA_ROOT, P, 'test')
    test_data = base.prepare_split(test_source, channel_scale)
    test_loader = DataLoader(
        base.ChannelDataset(test_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )
    test_metrics = base.evaluate(model, test_loader, device)
    gs_test_path = os.path.join(SOURCE_ROOT, 'gs_test_P10.json')
    with open(gs_test_path, 'r', encoding='utf-8') as file_handle:
        gs_test = json.load(file_handle)
    if v6.sha256_file(os.path.join(DATA_ROOT, 'test_P10.mat')) != gs_test.get('dataset_sha256'):
        raise ValueError('The saved GS test curve is for a different test dataset.')
    test_gap = {
        str(snr): float(
            test_metrics['nmse_db_by_snr'][str(snr)]
            - gs_test['nmse_db_by_snr'][str(snr)]
        ) for snr in SNR_GRID
    }
    result = {
        'P': P, 'seed': seed, 'source_model': source_file,
        'source_v6_best_epoch': source['result']['best_epoch'],
        'best_additional_epoch': best_epoch,
        'best_worst_validation_gap_to_gs_db': best_score,
        'selected_validation_dl_minus_gs_db_by_snr': history[best_epoch - 1][
            'validation_dl_minus_gs_db_by_snr'
        ] if best_epoch > 0 else initial_gaps,
        'test_metrics': test_metrics,
        'test_dl_minus_gs_db_by_snr': test_gap,
        'history': history,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'channel_scale': channel_scale,
        'number_of_layers': 40,
        'result': result,
    }, model_path)
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
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
    gs_validation = load_cached_gs_validation()

    summary_path = os.path.join(OUTPUT_ROOT, 'summary_unfolded_v7_p10_extended.json')
    results = []
    for seed in SEEDS:
        model_path = os.path.join(OUTPUT_ROOT, f'model_P10_seed{seed}_extended.pt')
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu', weights_only=False)
            results.append(saved['result'])
            print(f'Using completed extended model for seed {seed}.')
        else:
            started = time.time()
            result = run_seed(
                seed, train_data, validation_data, gs_validation,
                channel_scale, device,
            )
            result['seconds'] = time.time() - started
            results.append(result)
        with open(summary_path, 'w', encoding='utf-8') as file_handle:
            json.dump(results, file_handle, indent=2)
        result = results[-1]
        print(f"Seed {seed} test DL-GS gaps:",
              result['test_dl_minus_gs_db_by_snr'])

    print(f'Saved extended P=10 results to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
