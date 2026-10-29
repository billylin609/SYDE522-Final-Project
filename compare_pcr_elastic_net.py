'''
Elastic net (elastic_net_experiments.py) next to principal component regression (pcr_experiments.py),
on the same cells, split, CV folds and metrics.

The elastic net is fitted exactly as in elastic_net_experiments.py on each of its feature sets, and PCR is
fitted on the same feature sets, so every pair of rows compares the two models on identical inputs.
PCR on the Delta Q_{100-10}(V) curve, which has no elastic net counterpart, is added at the end.
Both predict log10(cycle life) and are tuned only on the 41 training cells by the same repeated 4-fold
CV (FOLDS), so the test cells are never used for any choice. Run from the project root:
    python compare_pcr_elastic_net.py
'''
import os

import matplotlib.pyplot as plt
import numpy as np

from elastic_net_experiments import FEATURE_SETS, L1_RATIOS, SPLITS, fit_elastic_net
from features import compute_features, load_raw, paper_split
from metrics.cycle_life import cycle_life_errors
from pcr_experiments import BLUE, GRID, ORANGE, fit_pcr, plot_pred_vs_obs


def elastic_net_cv_rmse(model):
    '''Mean per-fold CV RMSE of log10 life at the chosen alpha and l1_ratio: the same quantity, over the
    same folds, that GridSearchCV reports for PCR.'''
    net = model[-1]
    i = L1_RATIOS.index(net.l1_ratio_)
    j = np.flatnonzero(net.alphas_[i] == net.alpha_)[0]
    return np.sqrt(net.mse_path_[i, j]).mean()


def plot_comparison(errors, sets, path='plots/pcr_vs_elastic_net.png'):
    '''Mean percent error of both models on every shared feature set, one panel per split.'''
    pos = np.arange(len(sets))
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(18, 4.4), layout='constrained', sharey=True)
    for s, (ax, split) in enumerate(zip(axes, SPLITS)):
        ax.bar(pos - 0.2, [errors['ElasticNet', n][s][1] for n in sets], width=0.38, color=BLUE,
               label='Elastic net')
        ax.bar(pos + 0.2, [errors['PCR', n][s][1] for n in sets], width=0.38, color=ORANGE, label='PCR')
        ax.set_xticks(pos, [n.replace(' ', '\n') for n in sets])
        ax.set_title(split)
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean percent error (%)')
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Elastic net vs PCR on the same early-life features: mean percent error')
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


if __name__ == '__main__':
    raw = load_raw()
    F, names = compute_features(raw)
    dQ = raw['Qdlin100'] - raw['Qdlin10']  # (n_cells, 1000) Delta Q_{100-10}(V)
    life = raw['cycle_life']
    y = np.log10(life)
    train, test1, test2 = paper_split(raw['batch'])
    test1_clean = test1[life[test1] > 200]  # the paper's primary test without its 148-cycle cell
    eval_sets = [train, test1, test1_clean, test2]
    print(f'{len(train)} train, {len(test1)} primary test, {len(test2)} secondary test cells')

    # the elastic net's feature sets, columns picked as in elastic_net_experiments.py
    shared = {set_name: F[:, [names.index(f) for f in feats]]
              for set_name, feats in {**FEATURE_SETS, 'all features': names}.items()}

    results = {}  # (model, inputs) -> (fitted pipeline, X, CV RMSE of log10 life, chosen hyperparameters)
    for set_name, X in shared.items():
        model = fit_elastic_net(X[train], y[train])
        net = model[-1]
        results['ElasticNet', set_name] = (model, X, elastic_net_cv_rmse(model),
                                           f'alpha={net.alpha_:.3g} l1={net.l1_ratio_} '
                                           f'nz={np.count_nonzero(net.coef_)}/{X.shape[1]}')
        # every k up to the number of features; k = all of them is plain least squares
        model, _, cv_rmse, _ = fit_pcr(X[train], y[train], max_k=X.shape[1])
        results['PCR', set_name] = (model, X, cv_rmse.min(), f'k={model.named_steps["pca"].n_components}')
    # PCR on the whole Delta Q(V) curve, as in pcr_experiments.py
    for set_name, scale in (('dQ curve (scaled)', True), ('dQ curve (centered)', False)):
        model, _, cv_rmse, _ = fit_pcr(dQ[train], y[train], scale=scale)
        results['PCR', set_name] = (model, dQ, cv_rmse.min(), f'k={model.named_steps["pca"].n_components}')

    errors = {key: [cycle_life_errors(life[i], model.predict(X[i])) for i in eval_sets]
              for key, (model, X, _, _) in results.items()}

    header = f'{"model":11s}{"settings":28s}{"CV log":>7s} ' + ' '.join(f'{s:>15s}' for s in SPLITS)
    row = lambda errs: ' '.join(f'{r:6.0f} / {p:5.1f}%' for r, p in errs)
    print(f'\n== RMSE (cycles) / mean % error; CV log = training CV RMSE of log10 life ==\n'
          f'{"inputs":20s}{header}')
    last = None
    for (model_name, set_name), (_, _, cv, settings) in results.items():
        print(f'{set_name if set_name != last else "":20s}{model_name:11s}{settings:28s}{cv:7.4f} '
              f'{row(errors[model_name, set_name])}')
        last = set_name

    print('\n== Same inputs, head to head: sets where PCR has the lower mean % error ==')
    for s, split in enumerate(SPLITS):
        wins = [n for n in shared if errors['PCR', n][s][1] < errors['ElasticNet', n][s][1]]
        print(f'{split:11s} {len(wins)}/{len(shared)}  {", ".join(wins) or "-"}')

    # model choice uses training CV only, never the test sets
    print(f'\n== Best of each model across all its inputs, chosen by training CV ==\n{"inputs":20s}{header}')
    best = {}
    for model_name in ('ElasticNet', 'PCR'):
        best[model_name] = min((key for key in results if key[0] == model_name), key=lambda key: results[key][2])
        _, _, cv, settings = results[best[model_name]]
        print(f'{best[model_name][1]:20s}{model_name:11s}{settings:28s}{cv:7.4f} {row(errors[best[model_name]])}')

    os.makedirs('plots', exist_ok=True)
    plot_comparison(errors, list(shared))
    for model_name, path in (('ElasticNet', 'plots/compare_best_elastic_net.png'),
                             ('PCR', 'plots/compare_best_pcr.png')):
        model, X, _, _ = results[best[model_name]]
        plot_pred_vs_obs(model, [('Train', X[train], life[train]),
                                 ('Primary test', X[test1], life[test1]),
                                 ('Secondary test', X[test2], life[test2])],
                         f'{model_name} on {best[model_name][1]}, chosen by training CV', path)
