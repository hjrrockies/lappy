"""Shared utilities for lappy benchmark notebooks."""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAINS_WITH_REFERENCE = {
    'rect', 'eq_tri', 'iso_right_tri', 'L_shape',
    'GWW1', 'GWW2', 'disk', 'disk_sector',
}

DOMAIN_DISPLAY_NAMES = {
    'rect':          'Rectangle (2×1)',
    'L_shape':       'L-shape',
    'iso_right_tri': 'Isosceles right triangle',
    'eq_tri':        'Equilateral triangle',
    'disk_sector':   'Disk sector',
    'GWW1':          'GWW domain 1',
    'GWW2':          'GWW domain 2',
    'disk':          'Disk',
    'reg_ngon':      'Regular hexagon',
    'chevron':       'Chevron',
}

STRATEGY_LABELS = {
    'uniform':                  'Uniform',
    'angle_weighted':           'Angle-weighted',
    'singular_only':            'Singular only',
    'singular_angle_weighted':  'Singular + angle-weighted',
}

# Fixed ordering for consistent legends
STRATEGY_ORDER = [
    'uniform', 'angle_weighted', 'singular_only', 'singular_angle_weighted'
]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ensure_benchmark_suite_on_path(results_dir):
    """Add benchmarks/ to sys.path if needed so benchmark_suite is importable."""
    benchmarks_dir = os.path.abspath(os.path.join(results_dir, '..'))
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    project_root = os.path.abspath(os.path.join(results_dir, '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_sweep(sweep_name, results_dir='../results'):
    """Load all results for a given sweep name.

    Globs for ``*_{sweep_name}_*.npz`` in *results_dir*, loads each file with
    ``load_result``, converts to a pandas DataFrame, and adds derived columns.

    Returns
    -------
    df : pd.DataFrame
        One row per result; includes all BenchmarkConfig fields plus
        ``wall_time``, ``n_basis_actual``, ``n_bdry_pts``, ``n_int_pts``,
        ``n_eigs_returned``, ``median_tension``, ``max_tension``,
        ``median_rel_error``, ``max_rel_error``, ``n_basis``, ``fb_fraction``.
    raw : list of BenchmarkResult
        Raw result objects in the same order as DataFrame rows (needed for
        per-eigenvalue array access).
    """
    _ensure_benchmark_suite_on_path(results_dir)
    from benchmark_suite.results import load_result, results_to_dataframe

    results_dir = os.path.abspath(results_dir)
    paths = sorted(glob.glob(os.path.join(results_dir, f'*_{sweep_name}_*.npz')))
    if not paths:
        raise FileNotFoundError(
            f"No results found for sweep '{sweep_name}' in {results_dir}"
        )
    raw = [load_result(p) for p in paths]
    df = results_to_dataframe(raw)

    # Derived columns
    df['n_basis'] = df['n_fb'] + df['n_fs']
    df['fb_fraction'] = np.where(df['n_basis'] > 0, df['n_fb'] / df['n_basis'], np.nan)

    return df, raw


# ---------------------------------------------------------------------------
# Colour / marker helpers
# ---------------------------------------------------------------------------

_TAB10 = plt.cm.tab10.colors  # 10 distinct colours


def domain_color_map(domain_names):
    """Return a dict mapping domain name → colour, consistent across calls."""
    # Fixed ordering so colours don't depend on which domains appear in a sweep
    ordered = [d for d in DOMAIN_DISPLAY_NAMES if d in domain_names]
    # Any extras not in DOMAIN_DISPLAY_NAMES get appended
    ordered += [d for d in domain_names if d not in ordered]
    return {d: _TAB10[i % 10] for i, d in enumerate(ordered)}


def strategy_marker_map(strategies=None):
    """Return a dict mapping strategy name → marker."""
    markers = ['o', 's', '^', 'D', 'v', 'P']
    if strategies is None:
        strategies = STRATEGY_ORDER
    return {s: markers[i % len(markers)] for i, s in enumerate(strategies)}


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def set_publication_style():
    """Apply a clean publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.titlesize':    11,
        'axes.labelsize':    11,
        'legend.fontsize':   9,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'figure.figsize':    (6, 4),
        'figure.dpi':        100,
        'savefig.dpi':       150,
        'savefig.bbox':      'tight',
    })


# ---------------------------------------------------------------------------
# Convenience plot helpers
# ---------------------------------------------------------------------------

def label_domain(domain_name):
    """Return display name for a domain, falling back to the raw name."""
    return DOMAIN_DISPLAY_NAMES.get(domain_name, domain_name)


def label_strategy(strategy_name):
    """Return display name for a FB strategy."""
    return STRATEGY_LABELS.get(strategy_name, strategy_name)


def proxy_label(domain_name, metric='max_rel_error'):
    """Return '(tension proxy)' suffix for domains without reference eigs."""
    if domain_name not in DOMAINS_WITH_REFERENCE:
        return ' (tension proxy)'
    return ''


def log_formatter(ax, axis='y'):
    """Apply tidy log-scale tick formatting to an axis."""
    import matplotlib.ticker as ticker
    fmt = ticker.LogFormatterSciNotation(base=10, labelOnlyBase=True)
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def annotate_default_rtol(ax, rtol=1e-12, color='gray'):
    """Draw a vertical dashed line marking the default regularisation tolerance."""
    ax.axvline(rtol, color=color, linestyle=':', linewidth=1.2,
               label=f'Default rtol={rtol:.0e}')


def accuracy_threshold_line(ax, threshold=1e-10, color='gray', label=None):
    """Draw a horizontal dashed line marking an accuracy threshold."""
    if label is None:
        label = f'Threshold {threshold:.0e}'
    ax.axhline(threshold, color=color, linestyle='--', linewidth=0.8,
               alpha=0.7, label=label)
