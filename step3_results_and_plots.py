"""
Step 3: Results visualization and summary tables.
Run AFTER both step1 and step2.

Produces:
  figures/fig1_accuracy_coverage.png   — main result
  figures/fig2_auroc_vs_size.png       — OOD detection
  figures/fig3_adversarial_entropy.png — THE key finding
  figures/fig4_ece_calibration.png     — calibration
  tables/summary_mean_std.csv          — all metrics mean±std
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

Path("figures").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

COLORS  = {'dnn':'#2196F3','bnn':'#4CAF50','mc_dropout':'#FF9800','deep_ensemble':'#9C27B0'}
LABELS  = {'dnn':'DNN','bnn':'BNN (BBB)','mc_dropout':'MC Dropout','deep_ensemble':'Deep Ensemble'}
MARKERS = {'dnn':'o','bnn':'s','mc_dropout':'^','deep_ensemble':'D'}
MODELS  = ['dnn','bnn','mc_dropout','deep_ensemble']

# ── Load data ─────────────────────────────────────────────────────────────────

with open("results/baseline_results.json") as f:
    raw = json.load(f)

df = pd.DataFrame([{k:v for k,v in r.items()
                    if k not in ['cov_curve','acc_curve']} for r in raw])

# Aggregate over seeds
metric_cols = ['accuracy','nll','brier','ece','auroc',
               'entropy_id','entropy_ood','conf_id','conf_ood',
               'accept_rate','acc_accepted']
metric_cols = [c for c in metric_cols if c in df.columns]

agg = (df.groupby(['model','train_size'])[metric_cols]
         .agg(['mean','std'])
         .reset_index())
agg.columns = ['model','train_size'] + [f'{m}_{s}' for m,s in agg.columns[2:]]

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n=== SUMMARY TABLE (mean ± std over seeds) ===\n")
print(f"{'Model':<18} {'n':>6} {'Acc':>8} {'ECE':>8} {'AUROC':>8} {'SelAcc':>8}")
print("─"*60)
for ts in sorted(df['train_size'].unique()):
    for mt in MODELS:
        row = agg[(agg['model']==mt) & (agg['train_size']==ts)]
        if row.empty: continue
        r = row.iloc[0]
        print(f"{LABELS.get(mt,mt):<18} {ts:>6} "
              f"{r['accuracy_mean']:>7.3f} "
              f"{r['ece_mean']:>7.3f} "
              f"{r['auroc_mean']:>7.3f} "
              f"{r['acc_accepted_mean']:>7.3f}")
    print()

agg.to_csv("tables/summary_mean_std.csv", index=False)
print("Full table saved to tables/summary_mean_std.csv")

# ── Fig 1: Accuracy-Coverage ──────────────────────────────────────────────────

plot_sizes = [s for s in [500, 2000, 10000] if s in df['train_size'].values]
if not plot_sizes:
    plot_sizes = sorted(df['train_size'].unique())[:3]

fig, axes = plt.subplots(1, len(plot_sizes),
                         figsize=(4.5*len(plot_sizes), 4.5),
                         sharey=True)
if len(plot_sizes) == 1: axes = [axes]
fig.suptitle('Accuracy–Coverage Tradeoff  |  MNIST → Fashion-MNIST',
             fontsize=13, y=1.02)

for ax, ts in zip(axes, plot_sizes):
    subset = [r for r in raw if r['train_size'] == ts]
    for mt in MODELS:
        curves = [r for r in subset if r['model'] == mt and 'acc_curve' in r]
        if not curves: continue
        grid = np.linspace(0, 1, 200)
        interp = [np.interp(grid, r['cov_curve'], r['acc_curve']) for r in curves]
        mu = np.mean(interp, 0); sd = np.std(interp, 0)
        ax.plot(grid, mu, color=COLORS[mt], lw=2,
                label=LABELS[mt], marker=MARKERS[mt],
                markevery=25, markersize=4)
        ax.fill_between(grid, mu-sd, mu+sd, alpha=0.15, color=COLORS[mt])

    ax.set_xlabel('Coverage (fraction accepted)')
    ax.set_title(f'n = {ts:,}')
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.02)
    if ax is axes[0]: ax.set_ylabel('Accuracy on accepted samples')
    ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/fig1_accuracy_coverage.png')
plt.savefig('figures/fig1_accuracy_coverage.pdf')
plt.close()
print("\nFig 1 saved.")

# ── Fig 2: AUROC vs train size ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
for mt in MODELS:
    rows = agg[agg['model']==mt].sort_values('train_size')
    if rows.empty or 'auroc_mean' not in rows.columns: continue
    ax.plot(rows['train_size'], rows['auroc_mean'],
            color=COLORS[mt], marker=MARKERS[mt], lw=2,
            markersize=6, label=LABELS[mt])
    ax.fill_between(rows['train_size'],
                    rows['auroc_mean'] - rows['auroc_std'],
                    rows['auroc_mean'] + rows['auroc_std'],
                    alpha=0.15, color=COLORS[mt])
ax.axhline(0.5, ls='--', color='gray', lw=1, label='Random')
ax.set_xscale('log')
ax.set_xlabel('Training set size')
ax.set_ylabel('OOD AUROC ↑')
ax.set_title('OOD Detection vs Training Set Size  (MNIST → Fashion-MNIST)')
ax.legend(); ax.set_ylim(0.4, 1.05)
plt.tight_layout()
plt.savefig('figures/fig2_auroc_vs_size.png')
plt.savefig('figures/fig2_auroc_vs_size.pdf')
plt.close()
print("Fig 2 saved.")

# ── Fig 3: Adversarial entropy — THE KEY FINDING ──────────────────────────────

adv_path = Path("results/adversarial_results.json")
if adv_path.exists():
    with open(adv_path) as f:
        adv = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('BNN Entropy Under Structured OOD — The Key Experiment',
                 fontsize=13)

    # Left: entropy spectrum
    ax = axes[0]
    labels_plot, means, stds, colors_plot = [], [], [], []

    label_map = {
        'mnist_id':     ('MNIST (ID)', '#2196F3'),
        'rotated_15':   ('Rot. 15°',   '#64B5F6'),
        'rotated_30':   ('Rot. 30°',   '#42A5F5'),
        'rotated_45':   ('Rot. 45°',   '#1E88E5'),
        'rotated_60':   ('Rot. 60°',   '#1565C0'),
        'rotated_90':   ('Rot. 90°',   '#0D47A1'),
        'fashion_ood':  ('Fashion',    '#F44336'),
        'fgsm_eps0.05': ('FGSM ε=0.05','#FF9800'),
        'fgsm_eps0.1':  ('FGSM ε=0.1', '#F57C00'),
        'fgsm_eps0.2':  ('FGSM ε=0.2', '#E65100'),
        'fgsm_eps0.3':  ('FGSM ε=0.3', '#BF360C'),
    }

    for key, (label, color) in label_map.items():
        if key in adv:
            labels_plot.append(label)
            means.append(adv[key]['mean'])
            stds.append(adv[key].get('std', 0))
            colors_plot.append(color)

    y_pos = range(len(labels_plot))
    ax.barh(y_pos, means, xerr=stds, color=colors_plot,
            alpha=0.8, capsize=4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_plot)
    ax.axvline(adv.get('mnist_id', {}).get('mean', 0),
               ls='--', color='blue', lw=1.5, alpha=0.5,
               label='ID baseline')
    ax.set_xlabel('BNN Predictive Entropy (nats)')
    ax.set_title('BNN Entropy Across OOD Spectrum')
    ax.legend()

    # Right: FGSM confidence breakdown
    ax = axes[1]
    epsilons = [eps for eps in [0.05, 0.1, 0.2, 0.3]
                if f'fgsm_eps{eps}' in adv]
    if epsilons:
        dnn_accs = [adv[f'fgsm_eps{e}']['dnn_accuracy_after_attack'] for e in epsilons]
        bnn_ents = [adv[f'fgsm_eps{e}']['mean'] for e in epsilons]
        pct_conf = [adv[f'fgsm_eps{e}'].get('bnn_pct_confident_wrong', 0)
                    for e in epsilons]

        x = np.arange(len(epsilons))
        w = 0.3
        ax.bar(x - w, dnn_accs,  width=w, label='DNN acc after attack',
               color='#2196F3', alpha=0.8)
        ax.bar(x,     bnn_ents,  width=w, label='BNN entropy (nats)',
               color='#4CAF50', alpha=0.8)
        ax.bar(x + w, pct_conf,  width=w, label='BNN conf>0.8 + wrong',
               color='#F44336', alpha=0.8)

        # ID entropy reference line
        ax.axhline(adv.get('mnist_id', {}).get('mean', 0),
                   ls='--', color='green', lw=1.5, alpha=0.7,
                   label=f"BNN ID entropy (baseline)")

        ax.set_xticks(x)
        ax.set_xticklabels([f'ε={e}' for e in epsilons])
        ax.set_xlabel('FGSM Perturbation Strength')
        ax.set_ylabel('Value')
        ax.set_title('Adversarial Attack: Does BNN Know It\'s Fooled?')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig3_adversarial_entropy.png')
    plt.savefig('figures/fig3_adversarial_entropy.pdf')
    plt.close()
    print("Fig 3 saved.")
else:
    print("Fig 3 skipped — run step2 first.")

# ── Fig 4: ECE calibration ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
for mt in MODELS:
    rows = agg[agg['model']==mt].sort_values('train_size')
    if rows.empty or 'ece_mean' not in rows.columns: continue
    ax.plot(rows['train_size'], rows['ece_mean'],
            color=COLORS[mt], marker=MARKERS[mt], lw=2,
            markersize=6, label=LABELS[mt])
    ax.fill_between(rows['train_size'],
                    rows['ece_mean'] - rows['ece_std'],
                    rows['ece_mean'] + rows['ece_std'],
                    alpha=0.15, color=COLORS[mt])
ax.set_xscale('log')
ax.set_xlabel('Training set size')
ax.set_ylabel('ECE ↓ (lower = better calibrated)')
ax.set_title('Calibration vs Training Set Size')
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig4_ece_calibration.png')
plt.savefig('figures/fig4_ece_calibration.pdf')
plt.close()
print("Fig 4 saved.")

# ── Final print ───────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("OUTPUTS")
print("="*60)
print("  figures/fig1_accuracy_coverage.png  — main result")
print("  figures/fig2_auroc_vs_size.png      — OOD detection")
print("  figures/fig3_adversarial_entropy.png — KEY FINDING")
print("  figures/fig4_ece_calibration.png    — calibration")
print("  tables/summary_mean_std.csv         — all metrics")
print("\nShare results/adversarial_results.json and the")
print("fig3 plot — those determine the paper direction.")
