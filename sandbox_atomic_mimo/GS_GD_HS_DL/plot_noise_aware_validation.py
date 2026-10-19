"""Plot paired development validation curves, never final-test curves."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent
fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
comparisons=[]
for ax,p in zip(axes,[10,30]):
    for folder,label in [('nonlinear_development_results','Direct MLP'),
                         ('unfolded_development_results','Learned unfolding'),
                         ('noise_aware_development_results','Noise-aware unfolding')]:
        with (root/folder/f'validation_by_snr_P{p}.csv').open() as f:
            rows=list(csv.DictReader(f))
        x=[float(r['SNRdB']) for r in rows]
        y=[float(r['ValidationNMSEdB']) for r in rows]
        ax.plot(x,y,marker='o',label=label)
        comparisons.extend(dict(P=p,Method=label,SNRdB=xx,ValidationNMSEdB=yy) for xx,yy in zip(x,y))
    ax.set_title(f'P = {p}')
    ax.set_xlabel('Aggregate user-field SNR (dB)')
    ax.grid(alpha=.25)
axes[0].set_ylabel('Validation ratio-of-sums NMSE (dB)')
axes[1].legend(fontsize=9)
fig.suptitle('Development validation only: N = 8, K = 3, aggregate RSR = 50 dB')
fig.tight_layout()
out=root/'noise_aware_development_results'
fig.savefig(out/'validation_comparison.png',dpi=180)
with (out/'validation_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
    writer.writeheader();writer.writerows(comparisons)
print('Saved development validation comparison; same validation channels across all models.')
