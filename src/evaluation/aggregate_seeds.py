"""
aggregate_seeds.py - DermaAI multi-seed aggregation

Turns per-run test results (one row per experiment named <arch>_<loss>_seed<N>) into
mean / sample-std / min / max per loss variant. With only a few seeds the std is a rough
indication of run-to-run spread, not a precise estimate.
"""
import re
import numpy as np
import pandas as pd

NAME_RE = re.compile(r'^(?P<arch>.+?)_(?P<loss>ce|weighted_ce|focal)_seed(?P<seed>\d+)$')


def build_per_run_table(comparison_df, per_class_metrics):
    rows = []
    for _, r in comparison_df.iterrows():
        m = NAME_RE.match(r['experiment'])
        if not m:
            continue
        pc = per_class_metrics[r['experiment']]
        rows.append({
            'experiment': r['experiment'],
            'loss': m.group('loss'),
            'seed': int(m.group('seed')),
            'accuracy': r['accuracy'],
            'macro_f1': r['macro_f1'],
            'macro_recall': r['macro_recall'],
            'mel_recall': pc['mel']['recall'],
            'df_recall': pc['df']['recall'],
        })
    return pd.DataFrame(rows).sort_values(['loss', 'seed']).reset_index(drop=True)


def summarize(per_run):
    metrics = ['accuracy', 'macro_f1', 'macro_recall', 'mel_recall', 'df_recall']
    out = []
    for loss, g in per_run.groupby('loss'):
        row = {'loss': loss, 'n_seeds': len(g)}
        for k in metrics:
            row[f'{k}_mean'] = g[k].mean()
            row[f'{k}_std'] = g[k].std(ddof=1) if len(g) > 1 else float('nan')
            row[f'{k}_min'] = g[k].min()
            row[f'{k}_max'] = g[k].max()
        out.append(row)
    return pd.DataFrame(out)


def format_summary(summary):
    lines = []
    for _, r in summary.iterrows():
        parts = [f"{k}: {r[k+'_mean']:.3f} +/- {r[k+'_std']:.3f} [{r[k+'_min']:.3f}-{r[k+'_max']:.3f}]"
                 for k in ['accuracy', 'macro_f1', 'macro_recall', 'mel_recall', 'df_recall']]
        lines.append(f"{r['loss']:<12} (n={int(r['n_seeds'])})  " + " | ".join(parts))
    return "\n".join(lines)
