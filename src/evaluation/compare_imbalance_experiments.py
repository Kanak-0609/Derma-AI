"""
compare_imbalance_experiments.py - DermaAI Part 5 Analysis

Extracts melanoma and dermatofibroma recall/F1 across the three
class-imbalance-handling experiments (unweighted CE, weighted CE, Focal Loss)
to directly answer: did imbalance handling improve recall on the clinically
important minority classes?
"""

import json


def compare_mel_df(per_class_metrics_path):
    with open(per_class_metrics_path, 'r') as f:
        per_class = json.load(f)

    rows = []
    for exp_name, classes in per_class.items():
        mel = classes['mel']
        df = classes['df']
        rows.append({
            'experiment': exp_name,
            'mel_recall': mel['recall'],
            'mel_f1': mel['f1'],
            'df_recall': df['recall'],
            'df_f1': df['f1'],
        })
    return rows


if __name__ == "__main__":
    rows = compare_mel_df('/kaggle/working/repo/reports/classification/imbalance_experiments/per_class_metrics.json')
    print(f"{'Experiment':<32} {'mel recall':<12} {'mel f1':<10} {'df recall':<12} {'df f1':<10}")
    print("-" * 76)
    for r in rows:
        print(f"{r['experiment']:<32} {r['mel_recall']:<12.3f} {r['mel_f1']:<10.3f} {r['df_recall']:<12.3f} {r['df_f1']:<10.3f}")
