# V12 Colab pilot

Open `DL_v12_cnn_pilot_colab.ipynb` in Colab and run its cells in order. Select a GPU runtime when available. The notebook writes and runs the self-contained `colab_unfolded_dl_v12_cnn_pilot.py`; no earlier Python scripts need uploading.

The notebook reads these existing files from `MyDrive`:

- `DL_dataset_RSR25dB/train_P10.mat`, `validation_P10.mat`, `train_P30.mat`, `validation_P30.mat`
- `unfolded_dl_v6_gs_gap_results/gs_validation_P10.json` and `gs_validation_P30.json`
- `unfolded_dl_v11_continue_v10_p10_results/model_P10_seed20261026_v11.pt`
- `unfolded_dl_v6_gs_gap_results/model_P30_seed20261026.pt`

The experiment has two arms at each pilot length:

1. `loss`: keep the selected source architecture and train with SNR-balanced batches. Within each SNR group the loss uses total error power divided by total target power, then averages log NMSE over the eight SNR groups.
2. `cnn`: use the same loss and add a shared learned residual 1D CNN across the eight antennas after every fifth unfolded layer. The CNN output starts at zero, so the initial predictions match the source checkpoint.

Each arm starts from the same source weights for its pilot length. The pilot uses one seed, at most 40 epochs per arm, and checkpoint selection minimizes the largest validation DL-minus-GS gap. This is a controlled pilot; expand to three seeds only after deciding which arm merits the compute.

Results and per-epoch resume files are written to `MyDrive/unfolded_dl_v12_cnn_pilot_results`. Re-running the execution cell resumes unfinished arms and reuses completed selected checkpoints. The script checks the SHA-256 of each validation `.mat` against its cached GS curve before training. It does not open the test split.

Negative DL-minus-GS in dB means lower NMSE than GS. The goal is a robust negative gap at every SNR for both P values. A small negative value close to zero is only a near tie; confirm promising candidates with more seeds and a new independent final test set.
