"""
Step 3: Compare old vs fixed results and generate final paper figures.
Run after both step1_fixed and step2_fixed.

Produces:
  figures_fixed/fig1_entropy_comparison.png  -- old vs fixed BNN entropy
  figures_fixed/fig2_auroc_vs_size.png       -- all 4 models vs train size
  figures_fixed/fig3_adversarial_ordering.png -- THE key finding
  figures_fixed/fig4_accuracy_coverage.png   -- selective classification
  tables_fixed/table_summary.csv
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'figure.dpi':      130,
})

Path("figures_fixed").mkdir(exist_ok=True)
Path("tables_fixed").mkdir(exist_ok=True)

COLORS  = {
    'dnn':            '#2196F3',
    'bnn':            '#4CAF50',
    'mc_dropout':     '#FF9800',
    'deep_ensemble':  '#9C27B0',
}
LABELS  = {
    'dnn':            'DNN',
    'bnn':            'BNN (BBB)',
    'mc_dropout':     'MC Dropout',
    'deep_ensemble':  'Deep Ensemble',
}
MARKERS = {'dnn':'o','bnn':'s','mc_dropout':'^','deep_ensemble':'D'}
MODELS  = ['dnn','bnn','mc_dropout','deep_ensemble']


def load_and_agg(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame([{k:v for k,v in r.items()
                        if k not in ['cov_curve','acc_curve']} for r in data])
    metric_cols = [c for c in df.columns
                   if c not in ['model','train_size','seed']]
    agg = (df.groupby(['model','train_size'])[metric_cols]
             .agg(['mean','std'])
             .reset_index())
    agg.columns = (['model','train_size'] +
                   [f'{m}_{s}' for m,s in agg.columns[2:]])
    return df, agg, data


# ── Fig 1: Old vs Fixed BNN entropy_id ────────────────────────────────────────

def fig1_entropy_comparison():
    """Show the improvement in BNN after fixes."""
    old_path = Path("results/baseline_results.json")
    new_path = Path("results_fixed/all_results.json")

    if not old_path.exists() or not new_path.exists():
        print("  Fig 1 skipped: need both old and new results")
        return

    with open(old_path) as f: old = json.load(f)
    with open(new_path) as f: new = json.load(f)

    sizes = [500, 2000, 10000]

    def bnn_entropy_id(data, sizes):
        out = []
        for s in sizes:
            rows = [r for r in data if r['model']=='bnn' and r['train_size']==s]
            out.append(np.mean([r['entropy_id'] for r in rows]))
        return out

    old_h = bnn_entropy_id(old, sizes)
    new_h = bnn_entropy_id(new, sizes)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sizes))
    w = 0.35
    bars1 = ax.bar(x - w/2, old_h, width=w, color='#F44336', alpha=0.8,
                   label='Original (broken)')
    bars2 = ax.bar(x + w/2, new_h, width=w, color='#4CAF50', alpha=0.8,
                   label='Fixed (lr=3e-4, 80ep)')

    ax.axhline(0.5, ls='--', color='black', lw=1.5, alpha=0.7,
               label='Target (< 0.5 nats)')
    ax.axhline(np.log(10), ls=':', color='gray', lw=1,
               label=f'Max entropy = ln(10) = {np.log(10):.2f}')

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'n={s}' for s in sizes])
    ax.set_ylabel('BNN entropy on MNIST (ID, nats)')
    ax.set_title('BNN Convergence: Before vs After Hyperparameter Fix\n'
                 '(lower = model is confident on familiar data)')
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures_fixed/fig1_entropy_comparison.png')
    plt.close()
    print("  Fig 1 saved.")


# ── Fig 2: AUROC vs train size, all models ─────────────────────────────────────

def fig2_auroc_vs_size():
    new_path = Path("results_fixed/all_results.json")
    if not new_path.exists():
        print("  Fig 2 skipped: no fixed results yet")
        return

    _, agg, _ = load_and_agg(new_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, metric, label, arrow in [
        (axes[0], 'auroc',    'OOD AUROC',                      'up'),
        (axes[1], 'ece',      'Expected Calibration Error (ECE)','down'),
    ]:
        for mt in MODELS:
            rows = agg[agg['model']==mt].sort_values('train_size')
            mean_col = f'{metric}_mean'
            std_col  = f'{metric}_std'
            if rows.empty or mean_col not in rows.columns: continue
            ax.plot(rows['train_size'], rows[mean_col],
                    color=COLORS[mt], marker=MARKERS[mt],
                    lw=2, markersize=6, label=LABELS[mt])
            ax.fill_between(rows['train_size'],
                            rows[mean_col] - rows[std_col],
                            rows[mean_col] + rows[std_col],
                            alpha=0.15, color=COLORS[mt])
        if metric == 'auroc':
            ax.axhline(0.5, ls='--', color='gray', lw=1, label='Random')
            ax.set_ylim(0.4, 1.05)
        ax.set_xscale('log')
        ax.set_xlabel('Training set size')
        ax.set_ylabel(f'{label} {"↑" if arrow=="up" else "↓"}')
        ax.set_title(f'{label} vs Training Set Size')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figures_fixed/fig2_auroc_ece.png')
    plt.close()
    print("  Fig 2 saved.")


# ── Fig 3: Adversarial entropy ordering — THE finding ─────────────────────────

def fig3_adversarial_ordering():
    adv_path = Path("results_fixed/adversarial_results_fixed.json")
    if not adv_path.exists():
        # Fall back to original
        adv_path = Path("results/adversarial_results.json")
    if not adv_path.exists():
        print("  Fig 3 skipped: no adversarial results")
        return

    with open(adv_path) as f:
        adv = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BNN Entropy Across OOD Spectrum: The Key Finding", fontsize=13)

    # Left: entropy bar chart across input types
    ax = axes[0]
    order = [
        ('mnist_id',     'MNIST\n(ID)',       '#2196F3'),
        ('fgsm_eps0.05', 'FGSM\nε=0.05',     '#FF9800'),
        ('fgsm_eps0.1',  'FGSM\nε=0.1',      '#F57C00'),
        ('fgsm_eps0.2',  'FGSM\nε=0.2',      '#E65100'),
        ('fgsm_eps0.3',  'FGSM\nε=0.3',      '#BF360C'),
        ('rotated_15',   'Rot.\n15°',         '#64B5F6'),
        ('rotated_45',   'Rot.\n45°',         '#1565C0'),
        ('rotated_90',   'Rot.\n90°',         '#0D47A1'),
        ('fashion_ood',  'Fashion\n(OOD)',    '#F44336'),
    ]

    keys_present = [item for item in order if item[0] in adv]
    labels_p = [item[1] for item in keys_present]
    means    = [adv[item[0]]['mean'] for item in keys_present]
    stds     = [adv[item[0]].get('std', 0) for item in keys_present]
    colors_p = [item[2] for item in keys_present]

    bars = ax.bar(range(len(labels_p)), means, yerr=stds,
                  color=colors_p, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(labels_p)))
    ax.set_xticklabels(labels_p, fontsize=9)
    ax.set_ylabel('BNN Predictive Entropy (nats)')
    ax.set_title('Entropy Ordering:\nAdversarials < Rotations < Semantic OOD')

    # Annotate the key gap
    if 'fgsm_eps0.3' in adv and 'fashion_ood' in adv:
        adv_idx = [i for i,k in enumerate(keys_present) if k[0]=='fgsm_eps0.3']
        ood_idx = [i for i,k in enumerate(keys_present) if k[0]=='fashion_ood']
        if adv_idx and ood_idx:
            ax.annotate('', xy=(ood_idx[0], adv['fashion_ood']['mean']),
                        xytext=(adv_idx[0], adv['fgsm_eps0.3']['mean']),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    # Right: FGSM DNN accuracy vs BNN entropy
    ax = axes[1]
    epsilons = [eps for eps in [0.05, 0.1, 0.2, 0.3]
                if f'fgsm_eps{eps}' in adv]
    if epsilons:
        dnn_accs = [adv[f'fgsm_eps{e}']['dnn_accuracy_after_attack'] for e in epsilons]
        bnn_ents = [adv[f'fgsm_eps{e}']['mean'] for e in epsilons]
        id_ent   = adv.get('mnist_id', {}).get('mean', 0)
        ood_ent  = adv.get('fashion_ood', {}).get('mean', 0)

        ax2 = ax.twinx()
        l1, = ax.plot(epsilons, dnn_accs, 'o-', color='#F44336',
                      lw=2, markersize=7, label='DNN accuracy (drops)')
        l2, = ax2.plot(epsilons, bnn_ents, 's-', color='#4CAF50',
                       lw=2, markersize=7, label='BNN entropy (rises slowly)')
        ax2.axhline(id_ent,  ls='--', color='#2196F3', lw=1.2,
                    label=f'BNN ID entropy ({id_ent:.2f})')
        ax2.axhline(ood_ent, ls='--', color='#F44336', lw=1.2,
                    label=f'BNN Fashion entropy ({ood_ent:.2f})')

        ax.set_xlabel('FGSM epsilon (perturbation strength)')
        ax.set_ylabel('DNN accuracy after attack', color='#F44336')
        ax2.set_ylabel('BNN predictive entropy (nats)', color='#4CAF50')
        ax.set_title('As DNN Collapses, BNN Remains\nModerately Uncertain')

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='lower left')
        ax2.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig('figures_fixed/fig3_adversarial_ordering.png')
    plt.close()
    print("  Fig 3 saved.")


# ── Fig 4: Accuracy-coverage curve ────────────────────────────────────────────

def fig4_accuracy_coverage():
    new_path = Path("results_fixed/all_results.json")
    if not new_path.exists():
        print("  Fig 4 skipped: no fixed results yet")
        return

    with open(new_path) as f:
        raw = json.load(f)

    plot_sizes = [s for s in [500, 2000, 10000]
                  if any(r['train_size']==s for r in raw)]

    fig, axes = plt.subplots(1, len(plot_sizes),
                             figsize=(4.5*len(plot_sizes), 4.5),
                             sharey=True)
    if len(plot_sizes) == 1: axes = [axes]
    fig.suptitle('Accuracy-Coverage Tradeoff  |  MNIST (trained) vs Fashion-MNIST (OOD)',
                 fontsize=12, y=1.02)

    for ax, ts in zip(axes, plot_sizes):
        for mt in MODELS:
            curves = [r for r in raw
                      if r['train_size']==ts and r['model']==mt
                      and 'acc_curve' in r]
            if not curves: continue
            grid = np.linspace(0, 1, 200)
            interp = [np.interp(grid, r['cov_curve'], r['acc_curve'])
                      for r in curves]
            mu = np.mean(interp, 0); sd = np.std(interp, 0)
            ax.plot(grid, mu, color=COLORS[mt], lw=2,
                    label=LABELS[mt], marker=MARKERS[mt],
                    markevery=25, markersize=4)
            ax.fill_between(grid, mu-sd, mu+sd, alpha=0.15, color=COLORS[mt])

        ax.set_xlabel('Coverage (fraction of samples accepted)')
        ax.set_title(f'n = {ts:,}')
        ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.02)
        if ax is axes[0]: ax.set_ylabel('Accuracy on accepted samples')
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig('figures_fixed/fig4_accuracy_coverage.png')
    plt.close()
    print("  Fig 4 saved.")


# ── Summary table ─────────────────────────────────────────────────────────────

def make_summary_table():
    new_path = Path("results_fixed/all_results.json")
    if not new_path.exists():
        return

    _, agg, _ = load_and_agg(new_path)

    print("\n=== FIXED RESULTS SUMMARY (mean over 3 seeds) ===\n")
    print(f"{'Model':<18} {'n':>6} {'Acc':>7} {'ECE':>7} "
          f"{'AUROC':>7} {'H_id':>7} {'H_ood':>8} {'Accept':>8}")
    print('-'*72)

    for ts in sorted(agg['train_size'].unique()):
        for mt in MODELS:
            row = agg[(agg['model']==mt) & (agg['train_size']==ts)]
            if row.empty: continue
            r = row.iloc[0]
            print(f"{LABELS[mt]:<18} {ts:>6} "
                  f"{r.get('accuracy_mean',0):>7.3f} "
                  f"{r.get('ece_mean',0):>7.3f} "
                  f"{r.get('auroc_mean',0):>7.3f} "
                  f"{r.get('entropy_id_mean',0):>7.3f} "
                  f"{r.get('entropy_ood_mean',0):>8.3f} "
                  f"{r.get('accept_rate_mean',0):>8.3f}")
        print()

    agg.to_csv("tables_fixed/summary.csv", index=False)
    print("Table saved to tables_fixed/summary.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating figures...\n")
    fig1_entropy_comparison()
    fig2_auroc_vs_size()
    fig3_adversarial_ordering()
    fig4_accuracy_coverage()
    make_summary_table()

    print("\nDone. Outputs in figures_fixed/ and tables_fixed/")
    print("\nKey things to check in the summary table:")
    print("  1. BNN entropy_id should be < 0.5 (was 1.1-1.5)")
    print("  2. BNN accept_rate should be > 0.50 (was 0.03-0.12)")
    print("  3. Deep Ensemble still likely best on AUROC -- that is fine, it is the story")
    print("  4. BNN adversarial entropy should be < Fashion entropy -- that is the finding")
