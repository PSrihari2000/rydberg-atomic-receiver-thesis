"""
Colab-ready noise-aware unfolded channel estimator.

The script trains separate models for P=10 and P=30 using the same RSR=25 dB
datasets as the MATLAB experiment. It uses 40 unfolded layers and eight
learned scalars per layer. Test data is loaded only for the final NMSE report.

In Colab, set DATA_ROOT to the folder containing train_P10.mat, validation_P10.mat,
test_P10.mat, and the corresponding P=30 files.
"""

import json
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# In Colab, mount Google Drive first if the data is stored there:
# from google.colab import drive
# drive.mount('/content/drive')
DATA_ROOT = '/content/drive/MyDrive/DL_dataset_RSR25dB'
OUTPUT_ROOT = '/content/drive/MyDrive/unfolded_dl_v2_results'
P_LIST = [10, 30]
SNR_GRID = [-5, 0, 5, 10, 15, 20, 25, 30]
NUMBER_OF_LAYERS = 40
NUMBER_OF_PARAMETERS_PER_LAYER = 8
BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 8
SEED = 20261023


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_mat_v73(filename, names):
    """Read MATLAB v7.3 arrays and convert them to sample-first layout."""
    with h5py.File(filename, 'r') as file_handle:
        result = {}
        for name in names:
            array = np.array(file_handle[name])
            # MATLAB v7.3 stores complex arrays in HDF5 as a compound dtype
            # with separate ``real`` and ``imag`` fields.  Convert that
            # representation back to NumPy complex values before using abs,
            # matrix products, or real/imaginary-part extraction.
            if array.dtype.names and {'real', 'imag'}.issubset(array.dtype.names):
                array = array['real'] + 1j * array['imag']
            if array.ndim == 3:
                # h5py exposes these MATLAB arrays as [samples, second, first].
                array = array.transpose(0, 2, 1)
            else:
                array = array.reshape(-1)
            result[name] = array
    return result


def load_split(data_root, P, split_name):
    filename = os.path.join(data_root, f'{split_name}_P{P}.mat')
    names = ['Y_data', 'G_data', 'S_data', 'B_data', 'sigma2_data', 'SNR_dB_data']
    return read_mat_v73(filename, names)


def compute_pilot_bound(S):
    """S has shape [samples, K, P]; output has shape [samples]."""
    gram = S @ np.conjugate(np.swapaxes(S, 1, 2))
    eigenvalues = np.linalg.eigvalsh(gram)
    return np.maximum(eigenvalues[:, -1].real, 1e-6).astype(np.float32)


def prepare_split(source, channel_scale):
    Y = source['Y_data'].astype(np.float32) / channel_scale
    G = source['G_data'].astype(np.complex64) / channel_scale
    S = source['S_data'].astype(np.complex64)
    B = source['B_data'].astype(np.complex64) / channel_scale
    sigma2 = source['sigma2_data'].astype(np.float32).reshape(-1) / (channel_scale ** 2)
    snr = source['SNR_dB_data'].astype(np.float32).reshape(-1)

    return {
        'y': Y,
        's_real': S.real.astype(np.float32),
        's_imag': S.imag.astype(np.float32),
        'b_real': B.real.astype(np.float32),
        'b_imag': B.imag.astype(np.float32),
        'target_real': G.real.astype(np.float32),
        'target_imag': G.imag.astype(np.float32),
        'noise_ratio': sigma2,
        'pilot_bound': compute_pilot_bound(S),
        'snr': snr,
    }


class ChannelDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.number_of_samples = data['y'].shape[0]

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        return {name: torch.from_numpy(value[index]) for name, value in self.data.items()}


class UnfoldedMagnitudeEstimator(nn.Module):
    """Physics-informed unfolded model with eight learned scalars per layer."""

    def __init__(self, number_of_layers=40):
        super().__init__()
        self.number_of_layers = number_of_layers
        # Columns: base step, base momentum, raw ridge, noise-step slope,
        # noise-momentum slope, data-fidelity gain, gradient gain, update gain.
        initial = torch.cat([
            torch.zeros(number_of_layers, 1),
            -torch.ones(number_of_layers, 1),
            -3.0 * torch.ones(number_of_layers, 1),
            torch.zeros(number_of_layers, 5),
        ], dim=1)
        self.parameters_per_layer = nn.Parameter(initial)

    def forward(self, batch):
        y = batch['y']
        s_real = batch['s_real']
        s_imag = batch['s_imag']
        b_real = batch['b_real']
        b_imag = batch['b_imag']
        noise_ratio = batch['noise_ratio'].view(-1, 1, 1)
        pilot_bound = batch['pilot_bound'].view(-1, 1, 1).clamp_min(1e-6)
        batch_size, N, P = y.shape
        K = s_real.shape[1]
        real_channel = torch.zeros(batch_size, N, K, device=y.device, dtype=y.dtype)
        imag_channel = torch.zeros_like(real_channel)
        previous_real_channel = torch.zeros_like(real_channel)
        previous_imag_channel = torch.zeros_like(real_channel)
        noise_level = torch.log1p(noise_ratio / pilot_bound)

        for layer_index in range(self.number_of_layers):
            parameters = self.parameters_per_layer[layer_index]
            step_size = 1.9 * torch.sigmoid(parameters[0] + parameters[3] * noise_level)
            momentum = 0.95 * torch.sigmoid(parameters[1] + parameters[4] * noise_level)
            ridge_weight = F.softplus(parameters[2])
            data_gain = 2.0 * torch.sigmoid(parameters[5])
            gradient_gain = 2.0 * torch.sigmoid(parameters[6])
            update_gain = 2.0 * torch.sigmoid(parameters[7])

            intermediate_real = real_channel + momentum * (real_channel - previous_real_channel)
            intermediate_imag = imag_channel + momentum * (imag_channel - previous_imag_channel)
            field_real = torch.bmm(intermediate_real, s_real) - torch.bmm(intermediate_imag, s_imag) + b_real
            field_imag = torch.bmm(intermediate_real, s_imag) + torch.bmm(intermediate_imag, s_real) + b_imag
            amplitude = torch.sqrt(field_real.square() + field_imag.square() + 1e-12)
            residual_factor = data_gain * (amplitude - y) / amplitude.clamp_min(1e-6)
            error_real = residual_factor * field_real
            error_imag = residual_factor * field_imag
            s_real_transpose = s_real.transpose(1, 2)
            s_imag_transpose = s_imag.transpose(1, 2)
            gradient_real = torch.bmm(error_real, s_real_transpose) + torch.bmm(error_imag, s_imag_transpose)
            gradient_imag = -torch.bmm(error_real, s_imag_transpose) + torch.bmm(error_imag, s_real_transpose)
            previous_real_channel = real_channel
            previous_imag_channel = imag_channel
            normalized_gradient_real = gradient_gain * gradient_real / pilot_bound
            normalized_gradient_imag = gradient_gain * gradient_imag / pilot_bound
            denominator = 1.0 + step_size * ridge_weight * noise_ratio / pilot_bound
            real_channel = (intermediate_real - update_gain * step_size * normalized_gradient_real) / denominator
            imag_channel = (intermediate_imag - update_gain * step_size * normalized_gradient_imag) / denominator

        return real_channel, imag_channel


def move_batch_to_device(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def nmse_loss(predicted_real, predicted_imag, target_real, target_imag):
    error_power = (predicted_real - target_real).square() + (predicted_imag - target_imag).square()
    target_power = target_real.square() + target_imag.square()
    return error_power.mean() / target_power.mean().clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_error = 0.0
    total_energy = 0.0
    snr_error = {float(snr): [0.0, 0.0] for snr in SNR_GRID}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_real, pred_imag = model(batch)
        target_real = batch['target_real']
        target_imag = batch['target_imag']
        error = ((pred_real - target_real).square() + (pred_imag - target_imag).square()).sum(dim=(1, 2))
        energy = (target_real.square() + target_imag.square()).sum(dim=(1, 2))
        total_error += error.sum().item()
        total_energy += energy.sum().item()
        for snr_value in SNR_GRID:
            mask = batch['snr'] == snr_value
            if mask.any():
                snr_error[float(snr_value)][0] += error[mask].sum().item()
                snr_error[float(snr_value)][1] += energy[mask].sum().item()
    result = {'nmse': total_error / max(total_energy, 1e-12)}
    result['nmse_db'] = 10.0 * np.log10(max(result['nmse'], 1e-12))
    result['nmse_db_by_snr'] = {}
    for snr_value, values in snr_error.items():
        if values[1] > 0:
            result['nmse_db_by_snr'][str(int(snr_value))] = 10.0 * np.log10(max(values[0] / values[1], 1e-12))
    return result


def train_one_P(P, device):
    train_source = load_split(DATA_ROOT, P, 'train')
    channel_scale = float(np.sqrt(np.mean(np.abs(train_source['G_data']) ** 2)))
    channel_scale = max(channel_scale, np.sqrt(np.finfo(np.float32).eps))
    validation_source = load_split(DATA_ROOT, P, 'validation')
    test_source = load_split(DATA_ROOT, P, 'test')
    train_data = prepare_split(train_source, channel_scale)
    validation_data = prepare_split(validation_source, channel_scale)
    test_data = prepare_split(test_source, channel_scale)
    train_loader = DataLoader(ChannelDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(ChannelDataset(validation_data), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(ChannelDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = UnfoldedMagnitudeEstimator(NUMBER_OF_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_file = os.path.join(OUTPUT_ROOT, f'checkpoint_P{P}_unfolded_v2.pt')
    best_validation_nmse = np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_validation_nmse = checkpoint['best_validation_nmse']
        best_state = checkpoint['best_state']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming P={P} from epoch {start_epoch}.')
    for epoch in range(start_epoch, NUMBER_OF_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_real, pred_imag = model(batch)
            loss = nmse_loss(pred_real, pred_imag, batch['target_real'], batch['target_imag'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        validation_metrics = evaluate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train_loss': running_loss / max(len(train_loader), 1), 'validation_nmse_db': validation_metrics['nmse_db']})
        print(f'P={P}, epoch={epoch:02d}, train loss={history[-1]["train_loss"]:.6f}, validation NMSE={validation_metrics["nmse_db"]:.3f} dB')
        if validation_metrics['nmse'] < best_validation_nmse:
            best_validation_nmse = validation_metrics['nmse']
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_validation_nmse': best_validation_nmse,
            'best_state': best_state,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
        }, checkpoint_file)

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    output_file = os.path.join(OUTPUT_ROOT, f'model_P{P}_unfolded_v2.pt')
    torch.save({'model_state_dict': model.state_dict(), 'channel_scale': channel_scale, 'number_of_layers': NUMBER_OF_LAYERS, 'parameters_per_layer': NUMBER_OF_PARAMETERS_PER_LAYER, 'history': history, 'test_metrics': test_metrics}, output_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    return {'P': P, 'channel_scale': channel_scale, 'history': history, 'test_metrics': test_metrics, 'model_file': output_file}


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    results = []
    for P in P_LIST:
        started = time.time()
        result = train_one_P(P, device)
        result['seconds'] = time.time() - started
        results.append(result)
        print(f'P={P} final test NMSE = {result["test_metrics"]["nmse_db"]:.3f} dB')
        print('Test NMSE by SNR:', result['test_metrics']['nmse_db_by_snr'])
    with open(os.path.join(OUTPUT_ROOT, 'summary_unfolded_v2.json'), 'w', encoding='utf-8') as file_handle:
        json.dump(results, file_handle, indent=2)


if __name__ == '__main__':
    main()
