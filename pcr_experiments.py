'''
Principal component regression (PCR) on the same data, split and metrics as elastic_net_experiments.py.

PCR = standardize -> PCA (keep k components) -> ordinary least squares on the k component scores.
Like elastic net, it is a linear model that copes with many correlated inputs, but it does so by
compressing the inputs instead of shrinking the coefficients. PCA never looks at cycle life, so the
components kept are the ones with the most variance, not necessarily the most predictive ones.

Three experiments, all predicting log10(cycle life) and tuned only on the 41 training cells:
    1. 'dQ curve (scaled)'     PCA on the full Delta Q_{100-10}(V) curve, 1000 voltage points, each
                               voltage point standardized (what pcr_battery.py did).
    2. 'dQ curve (centered)'   same curve, only mean-centred. Every point is in Ah, so no rescaling is
                               needed, and quiet voltage regions are not amplified into noise.
    3. 'all features'          PCA on the 17 hand-made features of features.py, which is the direct
                               comparison with the elastic net 'all features' row.

k is chosen by the same repeated 4-fold CV on the training cells as the elastic net (FOLDS), so the
test cells are never used for any choice. Run from the project root:  python pcr_experiments.py
'''
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from elastic_net_experiments import FOLDS, SPLITS, fit_elastic_net
from features import compute_features, load_raw, paper_split
from metrics.cycle_life import cycle_life_errors

BLUE, ORANGE, INK, GRID = '#2a78d6', '#eb6834', '#555555', '#e0e0e0'
VOLTAGE = np.linspace(3.6, 2.0, 1000)  # voltage grid of Qdlin (1000 points, high -> low)
MAX_K = 15                             # largest number of components tried


def make_pcr(scale=True):
    '''Pipeline: (standardize or only centre) -> PCA -> least squares. k is set later by grid search.'''
    return Pipeline([
        # with_std=False only subtracts the mean; PCA centres anyway, so this keeps the pipeline shape
        ('scale', StandardScaler(with_std=scale)),
        ('pca', PCA()),
        ('reg', LinearRegression()),
    ])


def fit_pcr(X, y, scale=True, max_k=MAX_K):
    '''Choose k by the same repeated 4-fold CV on the training cells as the elastic net, refit on all of them.'''
    ks = list(range(1, min(max_k, X.shape[1]) + 1))
    search = GridSearchCV(make_pcr(scale), {'pca__n_components': ks},
                          cv=FOLDS, scoring='neg_root_mean_squared_error')
    search.fit(X, y)
    # mean CV RMSE of log10(cycle life) for every k, for the "choosing k" plot
    cv_rmse = -search.cv_results_['mean_test_score']
    # cumulative explained variance of the first max k components, same preprocessing, training cells
    full = PCA(n_components=len(ks)).fit(StandardScaler(with_std=scale).fit_transform(X))
    explained = np.cumsum(full.explained_variance_ratio_)
    return search.best_estimator_, ks, cv_rmse, explained


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def plot_choosing_k(curves, path='plots/pcr_choosing_k.png'):
    '''CV error vs k next to explained variance: high variance explained is not the same as good prediction.'''
    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.2), layout='constrained')
    for ax, (name, (model, ks, cv_rmse, explained)) in zip(axes, curves.items()):
        best_k = model.named_steps['pca'].n_components
        ax.plot(ks, cv_rmse, 'o-', color=BLUE, label='CV RMSE')
        ax.axvline(best_k, color=INK, ls=':', lw=1)
        ax.set(xlabel='Number of components k', ylabel='CV RMSE of log10(cycle life)',
               title=f'{name}: best k = {best_k}')
        style(ax)
        twin = ax.twinx()
        twin.plot(ks, explained, 's--', color=ORANGE, ms=4, label='Cumulative explained variance')
        twin.set_ylim(0, 1.02)
        twin.set_ylabel('Cumulative explained variance', color=ORANGE)
        twin.spines[['top']].set_visible(False)
    axes[0].legend(frameon=False, loc='upper right')
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


def plot_pred_vs_obs(model, sets, title, path='plots/pcr_pred_vs_obs.png'):
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4.6), layout='constrained', sharey=True)
    lo, hi = 100, 2400
    for ax, (label, X, life) in zip(axes, sets):
        rmse, mape = cycle_life_errors(life, model.predict(X))
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls='--')
        ax.scatter(life, 10 ** model.predict(X), s=36, color=BLUE, edgecolor='white', linewidth=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Observed cycle life')
        ax.set_title(f'{label}\nRMSE {rmse:.0f} cycles, error {mape:.1f}%')
        style(ax)
    axes[0].set_ylabel('Predicted cycle life')
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


def plot_loadings(model, title, path):
    '''Shape of each kept component along the voltage axis (sign of a component is arbitrary).'''
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    for i, comp in enumerate(model.named_steps['pca'].components_):
        ax.plot(VOLTAGE, comp, lw=1.4, label=f'PC{i + 1}')
    ax.invert_xaxis()  # discharge runs from high to low voltage
    ax.set(xlabel='Voltage (V)', ylabel='Loading', title=title)
    ax.legend(frameon=False, ncol=2)
    style(ax)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


def component_report(model, X, y, feature_names=None):
    '''For each kept component: share of input variance, correlation of its score with log10 life,
    and (for the hand-made features) the three features that weigh most in it.'''
    scores = model[:-1].transform(X)  # standardize + PCA, without the regression
    var = model.named_steps['pca'].explained_variance_ratio_
    for i in range(scores.shape[1]):
        r = np.corrcoef(scores[:, i], y)[0, 1]
        line = f'    PC{i + 1:<2d} variance {100 * var[i]:5.1f}%   corr with log10 life {r:+.2f}'
        if feature_names is not None:
            comp = model.named_steps['pca'].components_[i]
            top = np.argsort(-np.abs(comp))[:3]
            line += '   top: ' + ', '.join(f'{feature_names[j]} ({comp[j]:+.2f})' for j in top)
        print(line)


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

    # name: (inputs, standardize?, largest k tried). For the 17 features we try every k up to 17;
    # k = 17 keeps every component, which is the same as plain least squares on all features.
    experiments = {
        'dQ curve (scaled)': (dQ, True, MAX_K),
        'dQ curve (centered)': (dQ, False, MAX_K),
        'all features': (F, True, F.shape[1]),
    }

    header = ' '.join(f'{s:>15s}' for s in SPLITS)
    print(f'\n== RMSE (cycles) / mean % error ==\n{"model":23s}{"k":>3s} {header}')
    row = lambda errs: ' '.join(f'{r:6.0f} / {p:5.1f}%' for r, p in errs)

    fitted, results = {}, {}
    for name, (X, scale, max_k) in experiments.items():
        fitted[name] = fit_pcr(X[train], y[train], scale=scale, max_k=max_k)
        model = fitted[name][0]
        results[name] = [cycle_life_errors(life[i], model.predict(X[i])) for i in eval_sets]
        k = model.named_steps['pca'].n_components
        print(f'{"PCR " + name:23s}{k:3d} {row(results[name])}')

    # Elastic net on the same 17 features, so the main comparison sits in one table
    net = fit_elastic_net(F[train], y[train])
    errs = [cycle_life_errors(life[i], net.predict(F[i])) for i in eval_sets]
    print(f'{"ElasticNet all features":23s}{"":>3s} {row(errs)}')

    print('\n== What the kept components capture (training cells) ==')
    for name, (X, _, _) in experiments.items():
        print(f'{name}:')
        component_report(fitted[name][0], X[train], y[train],
                         feature_names=names if name == 'all features' else None)

    # model choice for the plots uses training CV only, never the test sets
    best_name = min(fitted, key=lambda n: fitted[n][2].min())
    best_model, best_X = fitted[best_name][0], experiments[best_name][0]
    print(f'\nBest by training CV: PCR on {best_name}')

    os.makedirs('plots', exist_ok=True)
    plot_choosing_k(fitted)
    plot_pred_vs_obs(best_model, [('Train', best_X[train], life[train]),
                                  ('Primary test', best_X[test1], life[test1]),
                                  ('Secondary test', best_X[test2], life[test2])],
                     f'PCR on {best_name}, k chosen by training CV')
    for name in ('dQ curve (scaled)', 'dQ curve (centered)'):
        tag = 'scaled' if 'scaled' in name else 'centered'
        plot_loadings(fitted[name][0], f'PCA loadings of ΔQ(V), {name}', f'plots/pcr_loadings_{tag}.png')
