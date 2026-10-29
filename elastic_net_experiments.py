import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import PAPER_MODELS, compute_features, load_raw, paper_split
from metrics.cycle_life import cycle_life_errors

BLUE, ORANGE, GRID = '#2a78d6', '#eb6834', '#e0e0e0'
L1_RATIOS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
FOLDS = RepeatedKFold(n_splits=4, n_repeats=10, random_state=0)  # 4-fold CV, repeated with random splits

FEATURE_SETS = {
    **PAPER_MODELS,
    # chosen by greedy forward selection in basis_experiments.py
    'forward selected': ['log var dQ', 'T integral 2-100', 'log|dQ(2V)|', 'Tmax max 2-100',
                         'log|skew dQ|', 'log|mean dQ|', 'log|min dQ|'],
}

# Severson et al. (2019) Table 1 as (RMSE cycles, mean % error) for
# train, primary test, primary test without the outlier cell, secondary test.
# Checked against Table 1 of the paper (Nature Energy 4, p. 386). The paper's 'full' model also leaves
# out four cells whose temperature probes lost contact (Fig. 3 caption); we keep them.
PAPER_TABLE1 = {
    'variance': [(103, 14.1), (138, 14.7), (138, 13.2), (196, 11.4)],
    'discharge': [(76, 9.8), (91, 13.0), (86, 10.1), (173, 8.6)],
    'full': [(51, 5.6), (118, 14.1), (100, 7.5), (214, 10.7)],
}
SPLITS = ['train', 'primary', 'prim. excl', 'secondary']


def fit_elastic_net(X, y):
    '''Standardize the inputs, then pick alpha and l1_ratio by repeated 4-fold CV on the training cells.'''
    model = make_pipeline(StandardScaler(),
                          ElasticNetCV(l1_ratio=L1_RATIOS, alphas=100, cv=FOLDS, max_iter=100_000))
    return model.fit(X, y)


def plot_vs_paper(ours, path='plots/elastic_net_vs_paper.png'):
    models = list(PAPER_TABLE1)
    pos = np.arange(len(models))
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(18, 4.4), layout='constrained', sharey=True)
    for s, (ax, split) in enumerate(zip(axes, SPLITS)):
        ax.bar(pos - 0.2, [ours[m][s][1] for m in models], width=0.38, color=BLUE, label='This work')
        ax.bar(pos + 0.2, [PAPER_TABLE1[m][s][1] for m in models], width=0.38, color=ORANGE,
               label='Severson et al. (Table 1)')
        ax.set_xticks(pos, models)
        ax.set_title(split)
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean percent error (%)')
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle('Elastic net on early-life features: mean percent error vs the paper')
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


if __name__ == '__main__':
    raw = load_raw()
    F, names = compute_features(raw)
    life = raw['cycle_life']
    y = np.log10(life)
    train, test1, test2 = paper_split(raw['batch'])
    test1_clean = test1[life[test1] > 200]  # the paper's primary test without its 148-cycle cell
    eval_sets = [train, test1, test1_clean, test2]
    print(f'{len(train)} train, {len(test1)} primary test, {len(test2)} secondary test cells')

    header = ' '.join(f'{s:>15s}' for s in SPLITS)
    print(f'\n== RMSE (cycles) / mean % error ==\n{"model":17s} {"source":6s} {header}')
    row = lambda errs: ' '.join(f'{r:6.0f} / {p:5.1f}%' for r, p in errs)

    dummy = DummyRegressor().fit(F[train], y[train])  # predicts the mean training log10 life
    print(f'{"dummy":17s} {"ours":6s} {row([cycle_life_errors(life[i], dummy.predict(F[i])) for i in eval_sets])}')

    ours, chosen = {}, {}
    for set_name, feats in {**FEATURE_SETS, 'all features': names}.items():
        cols = [names.index(f) for f in feats]
        model = fit_elastic_net(F[train][:, cols], y[train])
        ours[set_name] = [cycle_life_errors(life[i], model.predict(F[i][:, cols])) for i in eval_sets]
        print(f'{set_name:17s} {"ours":6s} {row(ours[set_name])}')
        if set_name in PAPER_TABLE1:
            print(f'{"":17s} {"paper":6s} {row(PAPER_TABLE1[set_name])}')
        chosen[set_name] = (model[-1], feats)

    print('\n== Elastic net hyperparameters and coefficients (standardized inputs) ==')
    for set_name, (net, feats) in chosen.items():
        print(f'{set_name}: alpha={net.alpha_:.3g}, l1_ratio={net.l1_ratio_}, '
              f'{np.count_nonzero(net.coef_)}/{len(feats)} non-zero')
        for f, c in sorted(zip(feats, net.coef_), key=lambda t: -abs(t[1])):
            if c != 0:
                print(f'    {f:22s} {c:+.4f}')

    os.makedirs('plots', exist_ok=True)
    plot_vs_paper(ours)
