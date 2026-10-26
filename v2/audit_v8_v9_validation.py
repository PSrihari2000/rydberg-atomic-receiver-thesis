"""Audit V8/V9 seed diversity and per-example validation behavior at 25/30 dB.

Loads the single float32 unfolded parameter tensor from each PyTorch zip
checkpoint using the Python standard library, then reproduces the estimator
and the V6 complex128 GS baseline with NumPy. No model training is performed.
"""
import csv
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'GS-GS-HS-DL' / 'DL_dataset_RSR25dB'
V6 = ROOT / 'unfolded_dl_v6_gs_gap_results'
V8 = ROOT / 'unfolded_dl_v8_continue_p10_results'
V9 = ROOT / 'unfolded_dl_v9_highsnr_finetune_p10_results'
SEEDS = (20261026, 20261027, 20261028)
SNR_POINTS = (25, 30)
N_LAYERS = 40
GS_MAX_ITERATIONS = 1000
GS_TOLERANCE = 1e-10
GS_BATCH_SIZE = 64


def load_mat_split(filename):
    wanted = ('Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data')
    result = {}
    with h5py.File(filename, 'r') as f:
        for name in wanted:
            a = np.array(f[name])
            if a.dtype.names and {'real', 'imag'}.issubset(a.dtype.names):
                a = a['real'] + 1j * a['imag']
            if a.ndim == 3:
                a = a.transpose(0, 2, 1)
            else:
                a = a.reshape(-1)
            result[name] = a
    return result


def checkpoint_parameters(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if '/data/' in n and not n.endswith('/')]
        if len(names) != 1:
            raise ValueError(f'Unexpected checkpoint tensor layout in {path}: {names}')
        blob = z.read(names[0])
    if len(blob) != N_LAYERS * 8 * 4:
        raise ValueError(f'Unexpected parameter tensor size in {path}: {len(blob)}')
    return np.frombuffer(blob, dtype='<f4').copy().reshape(N_LAYERS, 8)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def softplus(x):
    x = np.asarray(x, dtype=np.float32)
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x), dtype=np.float32))


def prepare(source, train_scale):
    y = (source['Y_data'] / train_scale).astype(np.float32)
    g = (source['G_data'] / train_scale).astype(np.complex64)
    s = source['S_data'].astype(np.complex64)
    b = (source['B_data'] / train_scale).astype(np.complex64)
    sigma2 = (source['sigma2_data'].astype(np.float32) / (train_scale ** 2)).reshape(-1)
    gram = s @ np.swapaxes(s.conj(), 1, 2)
    pilot_bound = np.maximum(np.linalg.eigvalsh(gram)[:, -1].real, 1e-6).astype(np.float32)
    return y, g, s, b, sigma2, pilot_bound


def predict(parameters, y, s, b, sigma2, pilot_bound, batch_size=64):
    output = np.empty((len(y), s.shape[1], s.shape[1]), dtype=np.complex64)
    # Here s.shape[1] is K; N is the observation-row count in y.
    output = np.empty((len(y), y.shape[1], s.shape[1]), dtype=np.complex64)
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        yr, sr, br = y[start:end], s[start:end], b[start:end]
        batch = end - start
        nr = sigma2[start:end].reshape(batch, 1, 1)
        pb = np.maximum(pilot_bound[start:end], 1e-6).reshape(batch, 1, 1)
        noise_level = np.log1p(nr / pb).astype(np.float32)
        x = np.zeros((batch, yr.shape[1], sr.shape[1]), dtype=np.complex64)
        previous = np.zeros_like(x)
        for row in parameters:
            step = 1.9 * sigmoid(row[0] + row[3] * noise_level)
            momentum = 0.95 * sigmoid(row[1] + row[4] * noise_level)
            ridge = softplus(row[2])
            data_gain = 2.0 * sigmoid(row[5])
            gradient_gain = 2.0 * sigmoid(row[6])
            update_gain = 2.0 * sigmoid(row[7])
            intermediate = x + momentum * (x - previous)
            field = intermediate @ sr + br
            amplitude = np.sqrt(field.real**2 + field.imag**2 + 1e-12).astype(np.float32)
            residual = data_gain * (amplitude - yr) / np.maximum(amplitude, 1e-6)
            err = residual * field
            grad = err @ np.swapaxes(sr.conj(), 1, 2)
            previous, x = x, (intermediate - update_gain * step * gradient_gain * grad / pb) / (
                1.0 + step * ridge * nr / pb
            )
        output[start:end] = x
    return output


def gs_estimate(y, s, b):
    count, n, _ = y.shape
    k = s.shape[1]
    st = s.transpose(0, 2, 1)
    a = np.concatenate((np.broadcast_to(st[:, None], (count, n, s.shape[2], k)), b[..., None]), axis=-1)
    matrix = np.einsum('bnpi,bnp,bnpj->bnij', a.conj(), y, a, optimize=True)
    _, vecs = np.linalg.eigh(matrix)
    principal = vecs[..., -1]
    projected = np.einsum('bnpi,bni->bnp', a, principal, optimize=True)
    radius = (np.abs(projected) * y).sum(axis=-1) / np.maximum(
        np.square(np.abs(projected)).sum(axis=-1), np.finfo(float).tiny
    )
    phase = np.exp(-1j * np.angle(principal[..., k]))
    channel = radius[..., None] * phase[..., None] * principal[..., :k]
    gram = s @ s.conj().transpose(0, 2, 1)
    right = s.conj().transpose(0, 2, 1) @ np.linalg.inv(gram)
    active = np.ones(count, dtype=bool)
    iterations = np.zeros(count, dtype=np.int32)
    for iteration in range(1, GS_MAX_ITERATIONS + 1):
        field = channel @ s + b
        updated = (y * np.exp(1j * np.angle(field)) - b) @ right
        change = np.square(np.abs(updated - channel)).sum(axis=(1, 2))
        energy = np.square(np.abs(updated)).sum(axis=(1, 2))
        converged = change / np.maximum(energy, np.finfo(float).eps) < GS_TOLERANCE
        channel = np.where(active[:, None, None], updated, channel)
        iterations[active] = iteration
        active &= ~converged
        if not active.any():
            break
    return channel, iterations


def main():
    train_file = DATA / 'train_P10.mat'
    val_file = DATA / 'validation_P10.mat'
    gs_val = json.loads((V6 / 'gs_validation_P10.json').read_text(encoding='utf-8'))
    v8_summary = json.loads((V8 / 'summary_unfolded_v8_p10_continuation.json').read_text(encoding='utf-8'))
    v9_summary = json.loads((V9 / 'summary_unfolded_v9_highsnr_p10.json').read_text(encoding='utf-8'))
    actual_val_hash = hashlib.sha256(val_file.read_bytes()).hexdigest()
    if actual_val_hash != gs_val['dataset_sha256']:
        raise ValueError('Workspace validation data does not match cached GS fingerprint.')

    train = load_mat_split(train_file)
    val = load_mat_split(val_file)
    channel_scale = float(np.sqrt(np.mean(np.abs(train['G_data']) ** 2)))
    y, g, s, b, sigma2, pilot_bound = prepare(val, channel_scale)
    snr = val['SNR_dB_data'].astype(float)
    cond = np.linalg.cond(s.astype(np.complex128))
    channel_norm = np.sqrt(np.square(np.abs(g)).sum(axis=(1, 2)))
    selected = np.flatnonzero(np.isin(snr, SNR_POINTS))

    records = []
    gs_iters_all = []
    for start in range(0, len(selected), GS_BATCH_SIZE):
        ix = selected[start:start + GS_BATCH_SIZE]
        gs_hat, iters = gs_estimate(
            val['Y_data'][ix].astype(np.complex128),
            val['S_data'][ix].astype(np.complex128),
            val['B_data'][ix].astype(np.complex128),
        )
        gs_iters_all.extend(zip(ix.tolist(), iters.tolist()))
        gs_iter_by_index = dict(zip(ix.tolist(), iters.tolist()))
        gs_err = np.square(np.abs(gs_hat - val['G_data'][ix])).sum(axis=(1, 2))
        target = np.square(np.abs(val['G_data'][ix])).sum(axis=(1, 2))
        for label, folder, suffix in (('V8', V8, 'v8'), ('V9', V9, 'v9')):
            for seed in SEEDS:
                p = checkpoint_parameters(folder / f'model_P10_seed{seed}_{suffix}.pt')
                pred = predict(p, y[ix], s[ix], b[ix], sigma2[ix], pilot_bound[ix])
                dl_err = np.square(np.abs(pred - g[ix])).sum(axis=(1, 2))
                dl_nmse = dl_err / np.maximum(np.square(np.abs(g[ix])).sum(axis=(1, 2)), 1e-30)
                gs_nmse = gs_err / np.maximum(target, 1e-30)
                gap = 10.0 * np.log10(np.maximum(dl_nmse, 1e-30)) - 10.0 * np.log10(np.maximum(gs_nmse, 1e-30))
                for j, sample_i in enumerate(ix):
                    records.append({
                        'sample_index': int(sample_i), 'SNR_dB': int(snr[sample_i]),
                        'model': label, 'seed': seed,
                        'dl_nmse_per_example': float(dl_nmse[j]),
                        'gs_nmse_per_example': float(gs_nmse[j]),
                        'dl_minus_gs_dB_per_example': float(gap[j]),
                        'channel_norm': float(channel_norm[sample_i]),
                        'pilot_matrix_condition': float(cond[sample_i]),
                        'pilot_bound': float(pilot_bound[sample_i]),
                        'gs_iterations': int(gs_iter_by_index[int(sample_i)]),
                    })

    out_csv = V9 / 'v8_v9_validation_highsnr_per_example.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for label in ('V8', 'V9'):
        saved_summary = v8_summary if label == 'V8' else v9_summary
        for seed in SEEDS:
            for point in SNR_POINTS:
                subset = [r for r in records if r['model'] == label and r['seed'] == seed and r['SNR_dB'] == point]
                gaps = np.array([r['dl_minus_gs_dB_per_example'] for r in subset])
                dl_energy = sum(r['dl_nmse_per_example'] * r['channel_norm']**2 for r in subset)
                gs_energy = sum(r['gs_nmse_per_example'] * r['channel_norm']**2 for r in subset)
                aggregate_gap = float(10 * np.log10(dl_energy / gs_energy))
                saved_seed = next(r for r in saved_summary if r['seed'] == seed)
                saved_gap_map = (saved_seed['selected_validation_dl_minus_gs_db_by_snr'])
                saved_gap = float(saved_gap_map[str(point)])
                summary.append({
                    'model': label, 'seed': seed, 'SNR_dB': point,
                    'samples': len(subset), 'fraction_examples_DL_better_than_GS': float(np.mean(gaps < 0)),
                    'mean_per_example_gap_dB': float(np.mean(gaps)),
                    'median_per_example_gap_dB': float(np.median(gaps)),
                    'p10_gap_dB': float(np.quantile(gaps, .1)),
                    'p90_gap_dB': float(np.quantile(gaps, .9)),
                    'aggregate_gap_dB_reproduced': aggregate_gap,
                    'aggregate_gap_dB_from_saved_summary': saved_gap,
                    'reproduction_delta_dB': aggregate_gap - saved_gap,
                    'GS_examples_at_iteration_cap': sum(r['gs_iterations'] == GS_MAX_ITERATIONS for r in subset),
                })
    summary_csv = V9 / 'v8_v9_validation_highsnr_summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print('validation_sha256_matches_GS_cache:', actual_val_hash == gs_val['dataset_sha256'])
    print('train_samples:', len(train['G_data']), 'validation_samples:', len(val['G_data']))
    print('GS_iterations_min/max_high_snr:', min(i for _, i in gs_iters_all), max(i for _, i in gs_iters_all))
    print('GS_examples_at_iteration_cap_by_snr:', {point: sum(1 for i, count in gs_iters_all if snr[i] == point and count == GS_MAX_ITERATIONS) for point in SNR_POINTS})
    print('wrote:', out_csv)
    print('wrote:', summary_csv)
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
