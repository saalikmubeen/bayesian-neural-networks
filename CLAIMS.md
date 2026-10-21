# Claims → evidence

Every number and claim in `report/bnn.tex`, with the file it comes from.
E1 = `results/E1_pilot/metrics.csv` (per seed; means via `python -m bnnstudy.evaluate results/E1_pilot` → `summary.txt`).

| # | Claim (as written in the report) | Value | Source file / command | Split / n | Checked |
|---|---|---|---|---|---|
| 1 | KL over-weighting factor N / n_batches | 62.5 = 2000 / 32 (ceil(2000/64) = 32; batch B = 64) | `bnnstudy/train.py` (`bbb_bug`: `kl_scale = 1 / n_batches`) | n = 2,000 | ☑ |
| 3 | Fixed BBB acc 0.936, ECE 0.022, NLL 0.230 | 0.9359, 0.0221, 0.2301 | E1, method `bbb` | MNIST test, 3 seeds | ☑ |
| 4 | Bug BBB acc 0.904, ECE 0.185, NLL 0.470 | 0.9037, 0.1852, 0.4702 | E1, `bbb_bug` | 3 seeds | ☑ |
| 5 | DNN acc 0.925, NLL 0.364, ECE 0.045, AUROC 0.857 / 0.872 | 0.9249, 0.3640, 0.0451, 0.8567, 0.8717 | E1, `dnn` | 3 seeds | ☑ |
| 6 | NLL temperature T = 2.05, NLL 0.256, ECE 0.008, AUROC 0.867 / 0.893 | 2.0480, 0.2560, 0.0075, 0.8666, 0.8928 | E1, `dnn_Tnll` | val-fitted | ☑ |
| 7 | Softness-matched T = 2.05, ECE 0.007, AUROC 0.866 / 0.893 | 2.0518, 0.0067, 0.8661, 0.8927 | E1, `dnn_Tmatch[bbb]` | val-fitted | ☑ |
| 8 | Fixed BBB AUROC 0.832 / 0.886 | 0.8316, 0.8856 | E1, `bbb` | 3 seeds | ☑ |
| 9 | Bug BBB AUROC 0.836 / 0.895 | 0.8355, 0.8951 | E1, `bbb_bug` | 3 seeds | ☑ |
| 12 | Paired diffs Fashion +0.026, −0.025, −0.104; KMNIST −0.010, −0.001, −0.010 | as written | E1, `bbb` − `dnn_Tmatch[bbb]` per seed | seeds 0,1,2 | ☑ |
| 13 | MI AUROC KMNIST: bug 0.945, fixed 0.895 | 0.9453, 0.8955 | E1, `auroc_kmnist_mi` | 3 seeds | ☑ |
| 14 | Fashion AUROC range 0.72–0.90 for the same method | bbb 0.7207–0.8955 | E1, `bbb` rows | seeds 0–2 | ☑ |
| 15 | Setup: 784-256-256-10, Adam 1e-3, 40 / 150 epochs, batch 64, 20-epoch warm-up, 50 MC samples, 5,000 val, 10,000 test/OOD, 15 ECE bins | config | `bnnstudy/train.py`, `models.py`, `data.py`, `metrics.py` | — | ☑ |
| 16 | E2 plan: n ∈ {500, 2,000, 10,000}, 10 seeds | config | `experiments/E2_mnist_grid/protocol.md` | — | ☑ |
| 17 | Pre-registered criterion 0.02; 1/N; N(0,1) | — | `experiments/E1_pilot/protocol.md` | — | ☑ |
| 18 | Raw BBB − DNN per seed: Fashion +0.043, −0.025, −0.093; KMNIST +0.013, +0.015, +0.013 (mean +0.014) | as written | E1, `bbb` − `dnn` | seeds 0–2 | ☑ |
| 19 | P2 equivalence: KMNIST mean Δ −0.007 (held), Fashion −0.034 (failed); seed spread up to 0.18 (0.7207–0.8955) | as written | E1; `experiments/E1_pilot/protocol.md` P2 | 3 seeds | ☑ |
| 20 | BBB accuracy lead 1.1 points; budgets 150 vs 40 epochs | 0.9359 − 0.9249 = 0.0110 | E1; `bnnstudy/train.py` METHODS | 3 seeds | ☑ |
| 21 | Matched T for bug BBB = 4.44 | 4.4437 | E1, `dnn_Tmatch[bbb_bug]` T | 3 seeds | ☑ |
| 22 | Bug BBB MI vs its matched DNN: KMNIST +0.026, +0.049, +0.032; Fashion mean −0.017 | as written | E1, `bbb_bug` `auroc_*_mi` − `dnn_Tmatch[bbb_bug]` `auroc_*_entropy` | seeds 0–2 | ☑ |
