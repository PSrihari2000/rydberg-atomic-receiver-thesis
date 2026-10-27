"""V13: direct shared-per-antenna MLP for magnitude-only channel estimation.

Trains separate P=10 and P=30 estimators on the RSR=25 dB dataset. For each
receive antenna, the network sees that antenna's normalized magnitude data and
background field, the known pilot matrix, and noise/pilot conditioning; it
predicts the three complex channel coefficients. One MLP is shared over the N
antenna rows. Training is supervised by channel NMSE with SNR-balanced batches.
Validation selects the checkpoint against the cached GS reference. Test data is
never loaded by this script.

In Colab, mount Drive at /content/gdrive and run this file after uploading it.
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
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = '/content/gdrive/MyDrive'
DATA_ROOT = os.path.join(ROOT, 'DL_dataset_RSR25dB')
GS_ROOT = os.path.join(ROOT, 'unfolded_dl_v6_gs_gap_results')
OUTPUT_ROOT = os.path.join(ROOT, 'unfolded_dl_v13_shared_mlp_results')
P_LIST = (10, 30)
SNR_GRID = (-5, 0, 5, 10, 15, 20, 25, 30)
SEED = 20261027
MAX_EPOCHS = 80
PATIENCE = 12
GROUP_BATCH = 16
BATCH_SIZE = GROUP_BATCH * len(SNR_GRID)
HIDDEN_WIDTH = 384
NUMBER_OF_RESIDUAL_BLOCKS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5


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
        arrays = {}
        names = ('Y_data', 'G_data', 'S_data', 'B_data',
                 'sigma2_data', 'SNR_dB_data')
        for name in names:
            value = np.asarray(handle[name])
            if value.dtype.names and {'real', 'imag'} <= set(value.dtype.names):
                value = value['real'] + 1j * value['imag']
            arrays[name] = value.transpose(0, 2, 1) if value.ndim == 3 else value.reshape(-1)
    return arrays


def prepare_split(raw, channel_scale):
    y = raw['Y_data'].astype(np.float32) / channel_scale
    g = raw['G_data'].astype(np.complex64) / channel_scale
    s = raw['S_data'].astype(np.complex64)
    b = raw['B_data'].astype(np.complex64) / channel_scale
    noise_ratio = raw['sigma2_data'].astype(np.float32).reshape(-1) / channel_scale ** 2
    snr = raw['SNR_dB_data'].astype(np.float32).reshape(-1)

    gram = s.astype(np.complex128) @ s.conj().astype(np.complex128).transpose(0, 2, 1)
    eigenvalues = np.linalg.eigvalsh(gram).real
    pilot_bound = np.maximum(eigenvalues[:, -1], 1e-6).astype(np.float32)
    condition = 0.5 * np.log(np.maximum(eigenvalues[:, -1], 1e-12) /
                             np.maximum(eigenvalues[:, 0], 1e-12))
    noise_feature = np.log1p(noise_ratio / pilot_bound)
    return {
        'y': y,
        's_real': s.real.astype(np.float32),
        's_imag': s.imag.astype(np.float32),
        'b_real': b.real.astype(np.float32),
        'b_imag': b.imag.astype(np.float32),
        'target_real': g.real.astype(np.float32),
        'target_imag': g.imag.astype(np.float32),
        'noise_feature': noise_feature.astype(np.float32),
        'pilot_condition': condition.astype(np.float32),
        'snr': snr,
    }


class ChannelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, index):
        return {name: torch.from_numpy(value[index]) for name, value in self.data.items()}


class BalancedSNRBatches(Sampler):
    def __init__(self, snr, seed):
        self.groups = [np.flatnonzero(snr == value) for value in SNR_GRID]
        if any(len(group) == 0 for group in self.groups):
            raise ValueError('Training split must contain all eight SNR groups.')
        if len({len(group) for group in self.groups}) != 1:
            raise ValueError('This sampler expects equal sample counts at each SNR.')
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.groups[0]) / GROUP_BATCH)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shuffled = [rng.permutation(group) for group in self.groups]
        order = rng.permutation(len(self))
        for batch_index in order:
            start = batch_index * GROUP_BATCH
            rows = [group[start:start + GROUP_BATCH] for group in shuffled]
            batch = np.concatenate(rows)
            yield rng.permutation(batch).tolist()


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.05)

    def forward(self, value):
        update = self.fc2(self.dropout(self.activation(self.fc1(self.norm(value)))))
        return value + 0.2 * update


class SharedAntennaMLP(nn.Module):
    """Separate P-specific model, shared over all receive-antenna rows."""
    def __init__(self, pilots, antennas=8, users=3):
        super().__init__()
        self.pilots = pilots
        self.antennas = antennas
        self.users = users
        input_width = 3 * pilots + 2 * users * pilots + 3
        self.input_layer = nn.Linear(input_width, HIDDEN_WIDTH)
        self.input_norm = nn.LayerNorm(HIDDEN_WIDTH)
        self.blocks = nn.Sequential(*[
            ResidualBlock(HIDDEN_WIDTH) for _ in range(NUMBER_OF_RESIDUAL_BLOCKS)
        ])
        self.output_norm = nn.LayerNorm(HIDDEN_WIDTH)
        self.output_layer = nn.Linear(HIDDEN_WIDTH, 2 * users)

    def forward(self, batch):
        y = batch['y']
        batch_size, antennas, pilots = y.shape
        if antennas != self.antennas or pilots != self.pilots:
            raise ValueError(f'Expected N={self.antennas}, P={self.pilots}; got N={antennas}, P={pilots}.')
        pilot_features = torch.cat((batch['s_real'].flatten(1), batch['s_imag'].flatten(1)), dim=1)
        pilot_features = pilot_features[:, None, :].expand(-1, antennas, -1)
        scalar_features = torch.stack((batch['noise_feature'], batch['pilot_condition'],
                                       (batch['snr'] - 12.5) / 17.5), dim=1)
        scalar_features = scalar_features[:, None, :].expand(-1, antennas, -1)
        features = torch.cat((y, batch['b_real'], batch['b_imag'],
                              pilot_features, scalar_features), dim=2)
        features = features.reshape(batch_size * antennas, -1)
        hidden = self.input_norm(self.input_layer(features))
        hidden = self.blocks(hidden)
        output = self.output_layer(self.output_norm(hidden)).reshape(batch_size, antennas, 2 * self.users)
        return output[..., :self.users], output[..., self.users:]


def move_to_device(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def balanced_log_nmse_loss(pred_real, pred_imag, batch):
    error = ((pred_real - batch['target_real']).square() +
             (pred_imag - batch['target_imag']).square()).sum(dim=(1, 2))
    energy = (batch['target_real'].square() + batch['target_imag'].square()).sum(dim=(1, 2))
    losses = []
    for snr_value in SNR_GRID:
        mask = batch['snr'] == snr_value
        if not mask.any():
            raise ValueError(f'Balanced batch is missing SNR={snr_value}.')
        losses.append(torch.log(error[mask].sum().clamp_min(1e-12) /
                                energy[mask].sum().clamp_min(1e-12)))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {str(value): [0.0, 0.0] for value in SNR_GRID}
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        pred_real, pred_imag = model(batch)
        error = ((pred_real - batch['target_real']).square() +
                 (pred_imag - batch['target_imag']).square()).sum(dim=(1, 2))
        energy = (batch['target_real'].square() + batch['target_imag'].square()).sum(dim=(1, 2))
        for snr_value in SNR_GRID:
            mask = batch['snr'] == snr_value
            totals[str(snr_value)][0] += error[mask].sum().item()
            totals[str(snr_value)][1] += energy[mask].sum().item()
    return {key: float(10 * np.log10(max(err / max(power, 1e-12), 1e-12)))
            for key, (err, power) in totals.items()}


def load_gs_reference(pilots, validation_path):
    path = os.path.join(GS_ROOT, f'gs_validation_P{pilots}.json')
    with open(path, 'r', encoding='utf-8') as handle:
        reference = json.load(handle)
    if sha256_file(validation_path) != reference['dataset_sha256']:
        raise ValueError(f'GS curve and validation data do not match for P={pilots}.')
    return reference['nmse_db_by_snr']


def model_state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_pilots(pilots, device):
    train_path = os.path.join(DATA_ROOT, f'train_P{pilots}.mat')
    validation_path = os.path.join(DATA_ROOT, f'validation_P{pilots}.mat')
    train_raw = load_split(pilots, 'train')
    channel_scale = float(np.sqrt(np.mean(np.abs(train_raw['G_data']) ** 2)))
    channel_scale = max(channel_scale, float(np.sqrt(np.finfo(np.float32).eps)))
    train_data = prepare_split(train_raw, channel_scale)
    del train_raw
    validation_data = prepare_split(load_split(pilots, 'validation'), channel_scale)
    gs_curve = load_gs_reference(pilots, validation_path)
    train_sampler = BalancedSNRBatches(train_data['snr'], SEED + pilots)
    train_loader = DataLoader(ChannelDataset(train_data), batch_sampler=train_sampler,
                              pin_memory=device.type == 'cuda')
    validation_loader = DataLoader(ChannelDataset(validation_data), batch_size=256,
                                   shuffle=False, pin_memory=device.type == 'cuda')

    set_seed(SEED + pilots)
    model = SharedAntennaMLP(pilots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    best_score = float('inf')
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    tag = f'model_P{pilots}_v13_shared_mlp.pt'
    resume_path = os.path.join(OUTPUT_ROOT, f'resume_P{pilots}_v13.pt')
    model_path = os.path.join(OUTPUT_ROOT, tag)
    summary_path = os.path.join(OUTPUT_ROOT, f'summary_P{pilots}_v13.json')
    start_epoch = 1
    if os.path.isfile(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved['channel_scale'] != channel_scale:
            raise ValueError('Resume checkpoint was made with a different channel scale.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        best_score, best_epoch = saved['best_score'], saved['best_epoch']
        best_state, stale_epochs = saved['best_state'], saved['stale_epochs']
        history, start_epoch = saved['history'], saved['next_epoch']
        print(f'P={pilots}: resuming at epoch {start_epoch}.', flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        train_sampler.epoch = epoch - 1
        running_loss = 0.0
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_real, pred_imag = model(batch)
            loss = balanced_log_nmse_loss(pred_real, pred_imag, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item()

        val_by_snr = evaluate(model, validation_loader, device)
        gaps = {key: val_by_snr[key] - float(gs_curve[key]) for key in val_by_snr}
        worst_gap = max(gaps.values())
        scheduler.step(worst_gap)
        history.append({'epoch': epoch,
                        'train_loss': running_loss / max(len(train_loader), 1),
                        'validation_nmse_db_by_snr': val_by_snr,
                        'validation_dl_minus_gs_db_by_snr': gaps,
                        'worst_validation_gap_to_gs_db': worst_gap})
        print(f'P={pilots}, epoch={epoch:03d}, train loss={history[-1]["train_loss"]:.6f}, '
              f'worst DL-GS validation gap={worst_gap:+.4f} dB', flush=True)

        if worst_gap < best_score:
            best_score, best_epoch = worst_gap, epoch
            best_state = model_state_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'channel_scale': channel_scale,
                    'best_score': best_score, 'best_epoch': best_epoch,
                    'best_state': best_state, 'stale_epochs': stale_epochs,
                    'history': history, 'next_epoch': epoch + 1}, resume_path)
        if stale_epochs >= PATIENCE:
            print(f'P={pilots}: early stopping after epoch {epoch}.', flush=True)
            break

    if best_state is None:
        raise RuntimeError(f'P={pilots}: no model checkpoint was selected.')
    model.load_state_dict(best_state)
    selected_val = evaluate(model, validation_loader, device)
    selected_gaps = {key: selected_val[key] - float(gs_curve[key]) for key in selected_val}
    result = {
        'P': pilots,
        'seed': SEED,
        'channel_scale': channel_scale,
        'architecture': 'shared per-antenna residual MLP',
        'hidden_width': HIDDEN_WIDTH,
        'residual_blocks': NUMBER_OF_RESIDUAL_BLOCKS,
        'best_epoch': best_epoch,
        'best_worst_validation_gap_db': max(selected_gaps.values()),
        'selected_validation_nmse_db_by_snr': selected_val,
        'selected_validation_dl_minus_gs_db_by_snr': selected_gaps,
        'validation_data_sha256': sha256_file(validation_path),
        'test_evaluated': False,
        'history': history,
    }
    torch.save({'model_state_dict': best_state, 'channel_scale': channel_scale,
                'pilots': pilots, 'snr_grid': SNR_GRID, 'result': result}, model_path)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(f'P={pilots} selected epoch={best_epoch}, worst validation gap='
          f'{result["best_worst_validation_gap_db"]:+.4f} dB; test not evaluated.', flush=True)
    return result


def check_files():
    missing = []
    for pilots in P_LIST:
        for split in ('train', 'validation'):
            path = os.path.join(DATA_ROOT, f'{split}_P{pilots}.mat')
            if not os.path.isfile(path):
                missing.append(path)
        gs_path = os.path.join(GS_ROOT, f'gs_validation_P{pilots}.json')
        if not os.path.isfile(gs_path):
            missing.append(gs_path)
    if missing:
        raise FileNotFoundError('Missing required Drive files:\n' + '\n'.join(missing))


def main():
    check_files()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('V13 shared-antenna MLP; device:', device, flush=True)
    results = []
    for pilots in P_LIST:
        started = time.time()
        result = train_pilots(pilots, device)
        result['seconds'] = time.time() - started
        results.append(result)
        with open(os.path.join(OUTPUT_ROOT, 'summary_v13.json'), 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
    print('Validation-only results saved to:', OUTPUT_ROOT, flush=True)
    print('Do not treat these results as final test performance.', flush=True)


if __name__ == '__main__':
    main()
