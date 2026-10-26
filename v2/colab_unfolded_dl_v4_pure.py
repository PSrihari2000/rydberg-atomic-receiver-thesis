"""Pure high-SNR-weighted unfolded DL v4 training.

This reuses the v3 physics-informed unfolded architecture and loss. It adds
depth and selects checkpoints by the mean of the eight per-SNR validation
NMSE values in dB, so every SNR contributes equally to model selection.
No GS initialization, Hankel projection, or classical-estimator output is used
inside this model.

Upload this file beside colab_unfolded_dl_v3_highsnr.py in Colab, mount Drive,
and run this file. The v3 source is imported for the model/data/loss functions.
"""

import copy
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


BASE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'colab_unfolded_dl_v3_highsnr.py')
if not os.path.isfile(BASE_SCRIPT):
    raise FileNotFoundError(
        'Upload colab_unfolded_dl_v3_highsnr.py beside this v4 script.'
    )

spec = importlib.util.spec_from_file_location('unfolded_dl_v3_base', BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# v4 experiment settings. P=10 and P=30 remain separate models.
base.OUTPUT_ROOT = '/content/gdrive/MyDrive/unfolded_dl_v4_pure_results'
base.NUMBER_OF_LAYERS = 80
base.NUMBER_OF_PARAMETERS_PER_LAYER = 8
base.BATCH_SIZE = 128
base.NUMBER_OF_EPOCHS = 100
base.LEARNING_RATE = 3e-4
base.PATIENCE = 15
base.SEED = 20261024


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_P(P, device):
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
        shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        base.ChannelDataset(validation_data), batch_size=base.BATCH_SIZE,
        shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        base.ChannelDataset(test_data), batch_size=base.BATCH_SIZE,
        shuffle=False, pin_memory=True
    )

    model = base.UnfoldedMagnitudeEstimator(base.NUMBER_OF_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE)
    checkpoint_file = os.path.join(
        base.OUTPUT_ROOT, f'checkpoint_P{P}_unfolded_v4_pure.pt'
    )
    best_validation_score = np.inf
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_validation_score = checkpoint['best_validation_score']
        best_state = checkpoint['best_state']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming P={P} from epoch {start_epoch}.')

    for epoch in range(start_epoch, base.NUMBER_OF_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_real, pred_imag = model(batch)
            loss = base.weighted_nmse_loss(
                pred_real, pred_imag,
                batch['target_real'], batch['target_imag'], batch['snr']
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        validation_metrics = base.evaluate(model, validation_loader, device)
        per_snr = validation_metrics['nmse_db_by_snr']
        missing_snr = [str(snr) for snr in base.SNR_GRID
                       if str(snr) not in per_snr]
        if missing_snr:
            raise ValueError(f'Validation data is missing SNR groups: {missing_snr}')

        # Macro-average in dB: each SNR point contributes equally to selection.
        validation_score = float(np.mean([
            per_snr[str(snr)] for snr in base.SNR_GRID
        ]))
        epoch_record = {
            'epoch': epoch,
            'train_loss': running_loss / max(len(train_loader), 1),
            'validation_nmse_db': validation_metrics['nmse_db'],
            'validation_macro_mean_nmse_db': validation_score,
            'validation_nmse_db_by_snr': per_snr,
        }
        history.append(epoch_record)
        print(
            f'P={P}, epoch={epoch:03d}/{base.NUMBER_OF_EPOCHS}, '
            f'train loss={epoch_record["train_loss"]:.6f}, '
            f'macro validation NMSE={validation_score:.3f} dB, '
            f'worst-SNR validation NMSE={max(per_snr.values()):.3f} dB'
        )

        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_validation_score': best_validation_score,
            'best_state': best_state,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
        }, checkpoint_file)

        if epochs_without_improvement >= base.PATIENCE:
            print(f'Early stopping P={P} at epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError(f'No valid checkpoint was selected for P={P}.')

    model.load_state_dict(best_state)
    test_metrics = base.evaluate(model, test_loader, device)
    output_file = os.path.join(
        base.OUTPUT_ROOT, f'model_P{P}_unfolded_v4_pure.pt'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'channel_scale': channel_scale,
        'number_of_layers': base.NUMBER_OF_LAYERS,
        'parameters_per_layer': base.NUMBER_OF_PARAMETERS_PER_LAYER,
        'selection_metric': 'mean per-SNR validation NMSE in dB',
        'best_validation_score': best_validation_score,
        'history': history,
        'test_metrics': test_metrics,
    }, output_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return {
        'P': P,
        'channel_scale': channel_scale,
        'best_validation_score': best_validation_score,
        'history': history,
        'test_metrics': test_metrics,
        'model_file': output_file,
    }


def main():
    os.makedirs(base.OUTPUT_ROOT, exist_ok=True)
    set_seed(base.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('Pure unfolded DL: 80 layers; no GS/Hankel operations in the model.')

    results = []
    for P in base.P_LIST:
        started = time.time()
        result = train_one_P(P, device)
        result['seconds'] = time.time() - started
        results.append(result)
        print(f'P={P} test NMSE = {result["test_metrics"]["nmse_db"]:.3f} dB')
        print('Test NMSE by SNR:', result['test_metrics']['nmse_db_by_snr'])

    summary_file = os.path.join(base.OUTPUT_ROOT,
                                'summary_unfolded_v4_pure.json')
    with open(summary_file, 'w', encoding='utf-8') as file_handle:
        json.dump(results, file_handle, indent=2)
    print(f'Saved v4 pure-DL results to: {base.OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
