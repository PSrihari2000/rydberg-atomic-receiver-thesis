"""Fine-tune pure unfolded DL using the worst validation gap to GS.

P=10 starts from the completed v3 high-SNR model. P=30 starts from the
completed v5 SNR-conditioned-only model. Three independent shuffle seeds
fine-tune each model. GS is computed on the same validation examples and is
used only as a checkpoint-selection reference; its estimate is never an
input or initialization for DL. The test split is evaluated after selection.

Upload this file beside the v3 and v5 Python scripts in Colab. The trained
v3 and v5 model files and the RSR25dB dataset must remain in MyDrive.
"""

import hashlib
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def import_neighbour(name, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Upload {filename} beside this v6 script.')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_neighbour('unfolded_v3', 'colab_unfolded_dl_v3_highsnr.py')
v5 = import_neighbour('unfolded_v5', 'colab_unfolded_dl_v5_snr_aware.py')

DATA_ROOT = '/content/gdrive/MyDrive/DL_dataset_RSR25dB'
OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v6_gs_gap_results'
SOURCE_MODELS = {
    10: '/content/gdrive/MyDrive/unfolded_dl_v3_highsnr_results/model_P10_unfolded_v2.pt',
    30: '/content/gdrive/MyDrive/unfolded_dl_v5_snr_aware_results/model_P30_snr_conditioned_only.pt',
}
P_LIST = (10, 30)
SEEDS = (20261026, 20261027, 20261028)
SNR_GRID = tuple(base.SNR_GRID)
NUMBER_OF_EPOCHS = 50
PATIENCE = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
GS_MAX_ITERATIONS = 1000
GS_TOLERANCE = 1e-10
GS_BATCH_SIZE = 128


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gs_batch(y, s, b):
    """Batched complex128 equivalent of MATLAB run_GS in GS_GD_HS_DL_final.m."""
    count, N, P = y.shape
    K = s.shape[1]
    s_t = s.transpose(0, 2, 1)
    a = np.concatenate(
        (np.broadcast_to(s_t[:, None], (count, N, P, K)), b[..., None]),
        axis=-1,
    )
    matrix = np.einsum('bnpi,bnp,bnpj->bnij', a.conj(), y, a, optimize=True)
    _, eigenvectors = np.linalg.eigh(matrix)
    principal = eigenvectors[..., -1]
    projected = np.einsum('bnpi,bni->bnp', a, principal, optimize=True)
    radius = (np.abs(projected) * y).sum(axis=-1) / np.maximum(
        np.square(np.abs(projected)).sum(axis=-1), np.finfo(float).tiny
    )
    phase = np.exp(-1j * np.angle(principal[..., K]))
    channel = radius[..., None] * phase[..., None] * principal[..., :K]

    gram = s @ s.conj().transpose(0, 2, 1)
    right = s.conj().transpose(0, 2, 1) @ np.linalg.inv(gram)
    active = np.ones(count, dtype=bool)
    for _ in range(GS_MAX_ITERATIONS):
        field = channel @ s + b
        updated = (y * np.exp(1j * np.angle(field)) - b) @ right
        change = np.square(np.abs(updated - channel)).sum(axis=(1, 2))
        energy = np.square(np.abs(updated)).sum(axis=(1, 2))
        converged = change / np.maximum(energy, np.finfo(float).eps) < GS_TOLERANCE
        channel = np.where(active[:, None, None], updated, channel)
        active &= ~converged
        if not active.any():
            break
    return channel


def gs_by_snr(source):
    """Use raw double-precision measurements, as in the MATLAB comparator."""
    y_all = source['Y_data']
    s_all = source['S_data']
    b_all = source['B_data']
    g_all = source['G_data']
    snr_all = source['SNR_dB_data'].reshape(-1)
    totals = {str(snr): [0.0, 0.0, 0] for snr in SNR_GRID}
    for start in range(0, len(y_all), GS_BATCH_SIZE):
        stop = min(start + GS_BATCH_SIZE, len(y_all))
        channel = gs_batch(
            np.asarray(y_all[start:stop], dtype=np.float64),
            np.asarray(s_all[start:stop], dtype=np.complex128),
            np.asarray(b_all[start:stop], dtype=np.complex128),
        )
        target = np.asarray(g_all[start:stop], dtype=np.complex128)
        error = np.square(np.abs(channel - target)).sum(axis=(1, 2))
        energy = np.square(np.abs(target)).sum(axis=(1, 2))
        for snr in SNR_GRID:
            selected = snr_all[start:stop] == snr
            totals[str(snr)][0] += float(error[selected].sum())
            totals[str(snr)][1] += float(energy[selected].sum())
            totals[str(snr)][2] += int(selected.sum())
        print(f'GS: {stop}/{len(y_all)} examples', flush=True)
    if any(totals[str(snr)][2] == 0 for snr in SNR_GRID):
        raise ValueError('One or more SNR groups have no examples.')
    return {
        'nmse_db_by_snr': {
            str(snr): float(10 * np.log10(
                totals[str(snr)][0] / totals[str(snr)][1]
            )) for snr in SNR_GRID
        },
        'sample_count_by_snr': {
            str(snr): totals[str(snr)][2] for snr in SNR_GRID
        },
    }


def cached_gs(P, split, filename):
    cache_file = os.path.join(OUTPUT_ROOT, f'gs_{split}_P{P}.json')
    fingerprint = sha256_file(filename)
    if os.path.isfile(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as file_handle:
            cached = json.load(file_handle)
        if cached.get('dataset_sha256') == fingerprint:
            print(f'Using GS {split} cache for P={P}.')
            return cached
    source = base.load_split(DATA_ROOT, P, split)
    print(f'Computing GS on {split} P={P}; this may take several minutes.')
    result = gs_by_snr(source)
    result.update({
        'P': P, 'split': split, 'dataset_sha256': fingerprint,
        'iterations_max': GS_MAX_ITERATIONS, 'change_tolerance': GS_TOLERANCE,
        'source': 'complex128 port of MATLAB run_GS; same saved split',
    })
    with open(cache_file, 'w', encoding='utf-8') as file_handle:
        json.dump(result, file_handle, indent=2)
    return result


def make_model(P, N, K, device):
    if P == 10:
        model = base.UnfoldedMagnitudeEstimator(40)
    else:
        model = v5.SNRConditionedUnfoldedEstimator(
            P, N, K, 40, learned_initializer=False,
            snr_conditioned_updates=True,
        )
    source_file = SOURCE_MODELS[P]
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f'Pretrained model missing: {source_file}')
    source = torch.load(source_file, map_location='cpu', weights_only=False)
    model.load_state_dict(source['model_state_dict'], strict=True)
    return model.to(device)


def train_one(P, seed, train_data, validation_data, gs_validation, device):
    set_seed(seed)
    train_loader = DataLoader(
        base.ChannelDataset(train_data), batch_size=BATCH_SIZE,
        shuffle=True, pin_memory=device.type == 'cuda',
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=device.type == 'cuda',
    )
    N, K = train_data['target_real'].shape[1:]
    model = make_model(P, N, K, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_file = os.path.join(OUTPUT_ROOT, f'checkpoint_P{P}_seed{seed}.pt')
    best_score = float('inf')
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint['best_score']
        best_state = checkpoint['best_state']
        best_epoch = checkpoint['best_epoch']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming P={P}, seed={seed}, epoch={start_epoch}.')

    for epoch in range(start_epoch, NUMBER_OF_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted_real, predicted_imag = model(batch)
            loss = base.weighted_nmse_loss(
                predicted_real, predicted_imag,
                batch['target_real'], batch['target_imag'], batch['snr'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        validation = base.evaluate(model, validation_loader, device)
        dl_by_snr = validation['nmse_db_by_snr']
        gs_by_snr_curve = gs_validation['nmse_db_by_snr']
        missing = [str(snr) for snr in SNR_GRID if str(snr) not in dl_by_snr]
        if missing:
            raise ValueError(f'Validation data lacks SNR groups: {missing}')
        gap = {
            str(snr): float(dl_by_snr[str(snr)] - gs_by_snr_curve[str(snr)])
            for snr in SNR_GRID
        }
        score = max(gap.values())
        record = {
            'epoch': epoch,
            'train_loss': running_loss / max(len(train_loader), 1),
            'validation_nmse_db_by_snr': dl_by_snr,
            'validation_dl_minus_gs_db_by_snr': gap,
            'worst_validation_gap_to_gs_db': score,
        }
        history.append(record)
        print(f'P={P} seed={seed} epoch={epoch:03d} '
              f'worst DL-GS gap={score:.3f} dB', flush=True)
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
            'best_score': best_score,
            'best_state': best_state,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
        }, checkpoint_file)
        if epochs_without_improvement >= PATIENCE:
            print(f'Early stopping P={P}, seed={seed}.')
            break

    if best_state is None:
        raise RuntimeError(f'No selected checkpoint for P={P}, seed={seed}.')
    model.load_state_dict(best_state)
    model_file = os.path.join(OUTPUT_ROOT, f'model_P{P}_seed{seed}.pt')
    return model, model_file, {
        'P': P, 'seed': seed, 'source_model': SOURCE_MODELS[P],
        'best_epoch': best_epoch, 'best_validation_gap_to_gs_db': best_score,
        'selected_validation_nmse_db_by_snr': history[best_epoch - 1]['validation_nmse_db_by_snr'],
        'selected_validation_dl_minus_gs_db_by_snr': history[best_epoch - 1]['validation_dl_minus_gs_db_by_snr'],
        'history': history, 'model_file': model_file,
    }


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    results = []
    summary_file = os.path.join(OUTPUT_ROOT, 'summary_unfolded_v6_gs_gap.json')
    for P in P_LIST:
        train_source = base.load_split(DATA_ROOT, P, 'train')
        channel_scale = float(np.sqrt(np.mean(np.abs(train_source['G_data']) ** 2)))
        channel_scale = max(channel_scale, np.sqrt(np.finfo(np.float32).eps))
        train_data = base.prepare_split(train_source, channel_scale)
        validation_source = base.load_split(DATA_ROOT, P, 'validation')
        validation_data = base.prepare_split(validation_source, channel_scale)
        gs_validation = cached_gs(
            P, 'validation', os.path.join(DATA_ROOT, f'validation_P{P}.mat')
        )
        for seed in SEEDS:
            model_file = os.path.join(OUTPUT_ROOT, f'model_P{P}_seed{seed}.pt')
            if os.path.isfile(model_file):
                saved = torch.load(model_file, map_location='cpu', weights_only=False)
                results.append(saved['result'])
                print(f'Using completed model P={P}, seed={seed}.')
                continue
            started = time.time()
            model, model_file, result = train_one(
                P, seed, train_data, validation_data, gs_validation, device
            )
            # The test split is first used after the validation checkpoint
            # has been selected and the selected weights loaded.
            test_source = base.load_split(DATA_ROOT, P, 'test')
            test_data = base.prepare_split(test_source, channel_scale)
            test_loader = DataLoader(
                base.ChannelDataset(test_data), batch_size=BATCH_SIZE,
                shuffle=False, pin_memory=device.type == 'cuda',
            )
            result['test_metrics'] = base.evaluate(model, test_loader, device)
            result['channel_scale'] = channel_scale
            result['seconds'] = time.time() - started
            torch.save({
                'model_state_dict': model.state_dict(),
                'P': P, 'seed': seed, 'channel_scale': channel_scale,
                'number_of_layers': 40, 'result': result,
            }, model_file)
            checkpoint_file = os.path.join(
                OUTPUT_ROOT, f'checkpoint_P{P}_seed{seed}.pt'
            )
            if os.path.isfile(checkpoint_file):
                os.remove(checkpoint_file)
            results.append(result)
            with open(summary_file, 'w', encoding='utf-8') as file_handle:
                json.dump(results, file_handle, indent=2)
            print(f"P={P}, seed={seed} test NMSE by SNR: "
                  f"{result['test_metrics']['nmse_db_by_snr']}")

        # Test GS is computed after every seed's checkpoint is selected.
        gs_test = cached_gs(
            P, 'test', os.path.join(DATA_ROOT, f'test_P{P}.mat')
        )
        for result in results:
            if result['P'] != P:
                continue
            dl_by_snr = result['test_metrics']['nmse_db_by_snr']
            result['test_dl_minus_gs_db_by_snr'] = {
                str(snr): float(
                    dl_by_snr[str(snr)] - gs_test['nmse_db_by_snr'][str(snr)]
                ) for snr in SNR_GRID
            }
        with open(summary_file, 'w', encoding='utf-8') as file_handle:
            json.dump(results, file_handle, indent=2)
    print(f'Saved v6 results to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
