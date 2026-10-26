"""V12 validation-only pilot: SNR-balanced loss, then a learned antenna CNN.

For P=10, transfer the selected V11 checkpoint; for P=30, transfer V6.
The loss arm changes only the training loss. The CNN arm uses the same loss
and adds a shared residual 1D CNN after every fifth unfolded update. No GS
estimate or Hankel projection enters either model. GS validation NMSE is
used only to choose checkpoints. No test split is loaded.

This file is self-contained for Colab. Run one pilot seed first; completed
experiments and epoch-level resumes are reused on a rerun.
"""

import hashlib
import json
import math
import os
import random
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = '/content/gdrive/MyDrive'
DATA_ROOT = os.path.join(ROOT, 'DL_dataset_RSR25dB')
GS_ROOT = os.path.join(ROOT, 'unfolded_dl_v6_gs_gap_results')
OUTPUT_ROOT = os.path.join(ROOT, 'unfolded_dl_v12_cnn_pilot_results')
SOURCE_PATHS = {
    10: os.path.join(ROOT, 'unfolded_dl_v11_continue_v10_p10_results',
                     'model_P10_seed20261026_v11.pt'),
    30: os.path.join(ROOT, 'unfolded_dl_v6_gs_gap_results',
                     'model_P30_seed20261026.pt'),
}
SNR_GRID = (-5, 0, 5, 10, 15, 20, 25, 30)
EXPERIMENTS = ((10, 'loss'), (10, 'cnn'), (30, 'loss'), (30, 'cnn'))
SEEDS = (20261026,)  # Pilot only; expand seeds after the architecture decision.
MAX_EPOCHS = 40
PATIENCE = 12
GROUP_BATCH = 16  # 16 from each SNR group: total batch size 128.
CNN_LEARNING_RATE = 3e-4
BACKBONE_LEARNING_RATE = 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_split(pilots, split):
    path = os.path.join(DATA_ROOT, f'{split}_P{pilots}.mat')
    with h5py.File(path, 'r') as handle:
        data = {}
        for name in ('Y_data', 'G_data', 'S_data', 'B_data',
                     'sigma2_data', 'SNR_dB_data'):
            value = np.asarray(handle[name])
            if value.dtype.names and {'real', 'imag'} <= set(value.dtype.names):
                value = value['real'] + 1j * value['imag']
            data[name] = value.transpose(0, 2, 1) if value.ndim == 3 else value.reshape(-1)
    return data


def raw_pilot_condition(pilots):
    gram = pilots.astype(np.complex128) @ pilots.conj().astype(np.complex128).transpose(0, 2, 1)
    eig = np.linalg.eigvalsh(gram).real
    return (0.5 * np.log(np.maximum(eig[:, -1], 1e-12) /
                         np.maximum(eig[:, 0], 1e-12))).astype(np.float32)


def prepare_split(raw, scale, condition_mean, condition_std):
    y = raw['Y_data'].astype(np.float32) / scale
    g = raw['G_data'].astype(np.complex64) / scale
    s = raw['S_data'].astype(np.complex64)
    b = raw['B_data'].astype(np.complex64) / scale
    gram = s @ s.conj().transpose(0, 2, 1)
    bound = np.maximum(np.linalg.eigvalsh(gram)[:, -1].real, 1e-6).astype(np.float32)
    condition = (raw_pilot_condition(s) - condition_mean) / condition_std
    return {
        'y': y, 's_real': s.real.astype(np.float32),
        's_imag': s.imag.astype(np.float32),
        'b_real': b.real.astype(np.float32),
        'b_imag': b.imag.astype(np.float32),
        'target_real': g.real.astype(np.float32),
        'target_imag': g.imag.astype(np.float32),
        'noise_ratio': (raw['sigma2_data'].astype(np.float32) / scale ** 2),
        'pilot_bound': bound,
        'pilot_condition': condition.astype(np.float32),
        'snr': raw['SNR_dB_data'].astype(np.float32),
    }


class ChannelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, index):
        return {key: torch.as_tensor(value[index]) for key, value in self.data.items()}


class BalancedBatches(Sampler):
    def __init__(self, snr, seed):
        self.groups = [np.flatnonzero(snr == point) for point in SNR_GRID]
        if any(len(group) == 0 for group in self.groups):
            raise ValueError('Every SNR group needs training samples.')
        if len(set(map(len, self.groups))) != 1:
            raise ValueError('This pilot expects equal training counts per SNR.')
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.groups[0]) / GROUP_BATCH)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = [rng.permutation(group) for group in self.groups]
        order = rng.permutation(len(self))
        for index in order:
            start = index * GROUP_BATCH
            batch = np.concatenate([group[start:start + GROUP_BATCH] for group in groups])
            yield rng.permutation(batch).tolist()


class ResidualAntennaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(14, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 6, 3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, real, imag, grad_real, grad_imag, snr, condition):
        n = real.shape[1]
        features = torch.cat((real, imag, grad_real, grad_imag), dim=2).transpose(1, 2)
        features = torch.cat((features, snr[:, None, None].expand(-1, 1, n),
                              condition[:, None, None].expand(-1, 1, n)), dim=1)
        delta = 0.1 * torch.tanh(self.layers(features)).transpose(1, 2)
        return delta[:, :, :3], delta[:, :, 3:]


class V12Estimator(nn.Module):
    def __init__(self, pilots, use_cnn):
        super().__init__()
        self.pilots = pilots
        self.use_cnn = use_cnn
        if pilots == 10:
            self.parameters_per_layer = nn.Parameter(torch.zeros(40, 8))
            self.condition_coefficients = nn.Parameter(torch.zeros(40, 6))
        elif pilots == 30:
            self.update_parameters = nn.Parameter(torch.zeros(40, 8))
            self.snr_slopes = nn.Parameter(torch.zeros(40, 8))
        else:
            raise ValueError('Only P=10 and P=30 are configured.')
        if use_cnn:
            self.cnn = ResidualAntennaCNN()

    def forward(self, batch):
        y = batch['y']
        sr, si = batch['s_real'], batch['s_imag']
        br, bi = batch['b_real'], batch['b_imag']
        noise = batch['noise_ratio'].view(-1, 1, 1)
        bound = batch['pilot_bound'].view(-1, 1, 1).clamp_min(1e-6)
        condition = batch['pilot_condition'].view(-1, 1, 1)
        snr = ((batch['snr'] - 12.5) / 17.5).clamp(-2.0, 2.0)
        real = y.new_zeros((len(y), y.shape[1], sr.shape[1]))
        imag = torch.zeros_like(real)
        previous_real, previous_imag = torch.zeros_like(real), torch.zeros_like(real)
        noise_level = torch.log1p(noise / bound)
        srt, sit = sr.transpose(1, 2), si.transpose(1, 2)

        for layer in range(40):
            if self.pilots == 10:
                p = self.parameters_per_layer[layer]
                c = self.condition_coefficients[layer]
                step = 1.9 * torch.sigmoid(p[0] + p[3] * noise_level + c[0] * condition)
                momentum = 0.95 * torch.sigmoid(p[1] + p[4] * noise_level + c[1] * condition)
                ridge = F.softplus(p[2] + c[2] * condition)
                data_gain = 2 * torch.sigmoid(p[5] + c[3] * condition)
                gradient_gain = 2 * torch.sigmoid(p[6] + c[4] * condition)
                update_gain = 2 * torch.sigmoid(p[7] + c[5] * condition)
            else:
                p = self.update_parameters[layer] + self.snr_slopes[layer] * snr[:, None]
                step = 1.9 * torch.sigmoid(p[:, 0]).view(-1, 1, 1)
                momentum = 0.95 * torch.sigmoid(p[:, 1]).view(-1, 1, 1)
                ridge = F.softplus(p[:, 2]).view(-1, 1, 1)
                data_gain = 2 * torch.sigmoid(p[:, 5]).view(-1, 1, 1)
                gradient_gain = 2 * torch.sigmoid(p[:, 6]).view(-1, 1, 1)
                update_gain = 2 * torch.sigmoid(p[:, 7]).view(-1, 1, 1)

            inter_real = real + momentum * (real - previous_real)
            inter_imag = imag + momentum * (imag - previous_imag)
            field_real = torch.bmm(inter_real, sr) - torch.bmm(inter_imag, si) + br
            field_imag = torch.bmm(inter_real, si) + torch.bmm(inter_imag, sr) + bi
            amplitude = torch.sqrt(field_real.square() + field_imag.square() + 1e-12)
            residual = data_gain * (amplitude - y) / amplitude.clamp_min(1e-6)
            error_real, error_imag = residual * field_real, residual * field_imag
            gradient_real = torch.bmm(error_real, srt) + torch.bmm(error_imag, sit)
            gradient_imag = -torch.bmm(error_real, sit) + torch.bmm(error_imag, srt)
            previous_real, previous_imag = real, imag
            norm_grad_real = gradient_gain * gradient_real / bound
            norm_grad_imag = gradient_gain * gradient_imag / bound
            denominator = 1.0 + step * ridge * noise / bound
            real = (inter_real - update_gain * step * norm_grad_real) / denominator
            imag = (inter_imag - update_gain * step * norm_grad_imag) / denominator
            if self.use_cnn and (layer + 1) % 5 == 0:
                delta_real, delta_imag = self.cnn(
                    real, imag, norm_grad_real, norm_grad_imag,
                    snr, batch['pilot_condition'],
                )
                real, imag = real + delta_real, imag + delta_imag

        return real, imag


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def balanced_log_nmse_loss(pred_real, pred_imag, batch):
    error = ((pred_real - batch['target_real']).square() +
             (pred_imag - batch['target_imag']).square()).sum(dim=(1, 2))
    energy = (batch['target_real'].square() + batch['target_imag'].square()).sum(dim=(1, 2))
    group_losses = []
    for snr in SNR_GRID:
        selected = batch['snr'] == snr
        if not selected.any():
            raise ValueError(f'Balanced batch lacks SNR={snr}.')
        group_losses.append(torch.log(error[selected].sum().clamp_min(1e-12) /
                                      energy[selected].sum().clamp_min(1e-12)))
    return torch.stack(group_losses).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {str(snr): [0.0, 0.0] for snr in SNR_GRID}
    for raw_batch in loader:
        batch = move(raw_batch, device)
        real, imag = model(batch)
        error = ((real - batch['target_real']).square() +
                 (imag - batch['target_imag']).square()).sum(dim=(1, 2))
        energy = (batch['target_real'].square() +
                  batch['target_imag'].square()).sum(dim=(1, 2))
        for snr in SNR_GRID:
            selected = batch['snr'] == snr
            totals[str(snr)][0] += error[selected].sum().item()
            totals[str(snr)][1] += energy[selected].sum().item()
    return {key: float(10 * np.log10(max(value[0] / value[1], 1e-12)))
            for key, value in totals.items()}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_reference(pilots):
    path = os.path.join(GS_ROOT, f'gs_validation_P{pilots}.json')
    with open(path, 'r', encoding='utf-8') as handle:
        curve = json.load(handle)
    data_path = os.path.join(DATA_ROOT, f'validation_P{pilots}.mat')
    if sha256_file(data_path) != curve['dataset_sha256']:
        raise ValueError(f'GS reference does not match {data_path}.')
    return curve['nmse_db_by_snr']


def check_files():
    required = list(SOURCE_PATHS.values())
    for pilots in (10, 30):
        required.extend([os.path.join(DATA_ROOT, f'{split}_P{pilots}.mat')
                         for split in ('train', 'validation')])
        required.append(os.path.join(GS_ROOT, f'gs_validation_P{pilots}.json'))
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError('Missing Drive files:\n' + '\n'.join(missing))


def run_arm(pilots, arm, seed, train_data, validation_data, source, gs_curve, device):
    set_seed(seed + pilots + (0 if arm == 'loss' else 1000))
    model = V12Estimator(pilots, arm == 'cnn').to(device)
    loaded = model.load_state_dict(source['model_state_dict'], strict=False)
    expected_missing = set()
    if arm == 'cnn':
        expected_missing = {'cnn.' + key for key in model.cnn.state_dict()}
    if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
        raise ValueError(f'Unexpected source transfer: {loaded}')

    validation_loader = DataLoader(ChannelDataset(validation_data), batch_size=128,
                                   shuffle=False, pin_memory=device.type == 'cuda')
    baseline = evaluate(model, validation_loader, device)
    initial_gaps = {key: baseline[key] - gs_curve[key] for key in baseline}
    old_gaps = source['result'].get('selected_validation_dl_minus_gs_db_by_snr')
    if old_gaps is None:
        raise ValueError('Source checkpoint lacks selected validation gaps.')
    discrepancy = max(abs(initial_gaps[key] - old_gaps[key]) for key in initial_gaps)
    if discrepancy > 0.005:
        raise ValueError(f'V12 does not reproduce source validation curve: {discrepancy:.6f} dB')
    print(f'P={pilots} arm={arm}: source curve reproduced within {discrepancy:.6g} dB', flush=True)

    sampler = BalancedBatches(train_data['snr'], seed + pilots)
    train_loader = DataLoader(ChannelDataset(train_data), batch_sampler=sampler,
                              pin_memory=device.type == 'cuda')
    backbone = [parameter for name, parameter in model.named_parameters()
                if not name.startswith('cnn.')]
    groups = [{'params': backbone, 'lr': BACKBONE_LEARNING_RATE}]
    if arm == 'cnn':
        groups.append({'params': model.cnn.parameters(), 'lr': CNN_LEARNING_RATE})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    best_score = max(initial_gaps.values())
    best_epoch = 0
    best_state = cpu_state(model)
    best_gaps = initial_gaps
    history = [{'epoch': 0, 'worst_validation_gap_db': best_score,
                'validation_gaps_db': initial_gaps}]
    stale = 0
    tag = f'P{pilots}_{arm}_seed{seed}'
    resume_path = os.path.join(OUTPUT_ROOT, f'resume_{tag}.pt')
    selected_path = os.path.join(OUTPUT_ROOT, f'model_{tag}.pt')
    start_epoch = 1
    if os.path.isfile(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved['source_path'] != SOURCE_PATHS[pilots]:
            raise ValueError('Resume source path differs from configured checkpoint.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        best_score, best_epoch = saved['best_score'], saved['best_epoch']
        best_state, best_gaps = saved['best_state'], saved['best_gaps']
        history, stale = saved['history'], saved['stale']
        start_epoch = saved['epoch'] + 1
        print(f'Resuming {tag} at epoch {start_epoch}.', flush=True)

    started = time.time()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        sampler.epoch = epoch
        model.train()
        loss_total = 0.0
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            real, imag = model(batch)
            loss = balanced_log_nmse_loss(real, imag, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {tag} epoch {epoch}.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()

        curve = evaluate(model, validation_loader, device)
        gaps = {key: curve[key] - gs_curve[key] for key in curve}
        score = max(gaps.values())
        scheduler.step(score)
        history.append({'epoch': epoch, 'train_log_nmse': loss_total / len(train_loader),
                        'validation_nmse_db': curve, 'validation_gaps_db': gaps,
                        'worst_validation_gap_db': score})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state, best_gaps = cpu_state(model), gaps
            stale = 0
        else:
            stale += 1
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_score': best_score, 'best_epoch': best_epoch,
                    'best_state': best_state, 'best_gaps': best_gaps,
                    'history': history, 'stale': stale,
                    'source_path': SOURCE_PATHS[pilots]}, resume_path)
        print(f'{tag} epoch={epoch:03d} worst={score:+.4f} dB '
              f'gap20={gaps["20"]:+.4f} gap25={gaps["25"]:+.4f} '
              f'gap30={gaps["30"]:+.4f} best={best_score:+.4f}', flush=True)
        if stale >= PATIENCE:
            break

    result = {'P': pilots, 'arm': arm, 'seed': seed, 'source_path': SOURCE_PATHS[pilots],
              'epochs_run': history[-1]['epoch'], 'best_epoch': best_epoch,
              'best_worst_validation_gap_db': best_score,
              'selected_validation_gaps_db': best_gaps,
              'initial_validation_gaps_db': initial_gaps,
              'seconds_this_session': time.time() - started,
              'test_evaluated': False}
    torch.save({'model_state_dict': best_state, 'channel_scale': source['channel_scale'],
                'condition_mean': source.get('condition_mean'),
                'condition_std': source.get('condition_std'),
                'result': result}, selected_path)
    with open(os.path.join(OUTPUT_ROOT, f'history_{tag}.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    return result


def main():
    check_files()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    summary_path = os.path.join(OUTPUT_ROOT, 'summary_v12_validation_pilot.json')
    results = []
    for pilots in (10, 30):
        source = torch.load(SOURCE_PATHS[pilots], map_location='cpu', weights_only=False)
        train_raw = load_split(pilots, 'train')
        train_scale = float(np.sqrt(np.mean(np.abs(train_raw['G_data']) ** 2)))
        if not np.isclose(train_scale, source['channel_scale'], rtol=1e-6):
            raise ValueError(f'P={pilots}: source channel scale does not match train set.')
        scale = float(source['channel_scale'])
        raw_condition = raw_pilot_condition(train_raw['S_data'])
        mean = float(source.get('condition_mean', raw_condition.mean()))
        std = float(source.get('condition_std', max(raw_condition.std(), 1e-6)))
        if pilots == 10:
            if abs(mean - raw_condition.mean()) > 1e-5 or abs(std - raw_condition.std()) > 1e-5:
                raise ValueError('P=10 pilot-condition statistics differ from V11.')
        train_data = prepare_split(train_raw, scale, mean, std)
        del train_raw
        validation_data = prepare_split(load_split(pilots, 'validation'), scale, mean, std)
        gs_curve = load_reference(pilots)
        for selected_pilots, arm in EXPERIMENTS:
            if selected_pilots != pilots:
                continue
            for seed in SEEDS:
                tag = f'P{pilots}_{arm}_seed{seed}'
                selected_path = os.path.join(OUTPUT_ROOT, f'model_{tag}.pt')
                if os.path.isfile(selected_path):
                    result = torch.load(selected_path, map_location='cpu',
                                        weights_only=False)['result']
                    print(f'Using completed {tag}.', flush=True)
                else:
                    result = run_arm(pilots, arm, seed, train_data, validation_data,
                                     source, gs_curve, device)
                results.append(result)
                with open(summary_path, 'w', encoding='utf-8') as handle:
                    json.dump(results, handle, indent=2)
    print('Saved V12 validation-only pilot results:', OUTPUT_ROOT, flush=True)
    print('The test split was not loaded or evaluated.', flush=True)


if __name__ == '__main__':
    main()
