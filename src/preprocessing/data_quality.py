"""
data_quality.py
DermaAI - Data Quality Module

Performs image-level and dataset-level quality checks on the HAM10000
classification dataset and produces:
    - reports/metadata_report.csv
    - reports/class_distribution.png
    - reports/data_quality_report.html

Usage (from a notebook or script):
    from data_quality import run_data_quality_checks
    run_data_quality_checks(meta_csv_path, output_dir)
"""

import os
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # safe for headless/notebook environments
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from datetime import datetime


def check_image_quality(image_paths):
    """Image-level checks: corruption, dimensions, mode, aspect ratio, content hash."""
    results = []
    for path in tqdm(image_paths, desc="Checking images"):
        record = {
            'path': path,
            'corrupted': False,
            'width': None,
            'height': None,
            'mode': None,
            'aspect_ratio': None,
            'file_hash': None,
            'error': None
        }
        try:
            with open(path, 'rb') as f:
                file_bytes = f.read()
                record['file_hash'] = hashlib.md5(file_bytes).hexdigest()

            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                w, h = img.size
                record['width'] = w
                record['height'] = h
                record['mode'] = img.mode
                record['aspect_ratio'] = round(w / h, 3)

        except Exception as e:
            record['corrupted'] = True
            record['error'] = str(e)

        results.append(record)

    return pd.DataFrame(results)


def find_duplicate_groups(quality_df, meta_df, path_col='image_path'):
    """Identify duplicate files by content hash, and report their split/lesion membership."""
    dup_counts = quality_df['file_hash'].value_counts()
    dup_hashes = dup_counts[dup_counts > 1].index.tolist()

    meta_indexed = meta_df.set_index(path_col)
    duplicate_report = []

    for h in dup_hashes:
        dup_paths = quality_df[quality_df['file_hash'] == h]['path'].tolist()
        entries = []
        for p in dup_paths:
            if p in meta_indexed.index:
                row = meta_indexed.loc[p]
                entries.append({
                    'path': p,
                    'image_id': row.get('image_id'),
                    'lesion_id': row.get('lesion_id'),
                    'split': row.get('split')
                })
        splits_involved = set(e['split'] for e in entries)
        duplicate_report.append({
            'hash': h,
            'entries': entries,
            'crosses_split_boundary': len(splits_involved) > 1
        })

    return duplicate_report


def analyze_dataset_level(meta_df):
    """Dataset-level checks: missing metadata, class imbalance, demographic distributions."""
    missing = meta_df.isnull().sum().to_dict()

    class_counts = meta_df['diagnosis'].value_counts()
    imbalance_ratio = float(class_counts.max() / class_counts.min())

    sex_counts = meta_df['sex'].value_counts(dropna=False).to_dict() if 'sex' in meta_df else {}
    localization_counts = meta_df['localization'].value_counts(dropna=False).to_dict() if 'localization' in meta_df else {}

    return {
        'missing_values': missing,
        'class_counts': class_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'sex_counts': sex_counts,
        'localization_counts': localization_counts,
    }


def plot_class_distribution(meta_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    meta_df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Overall Class Distribution (HAM10000)')
    axes[0].set_ylabel('Image count')
    axes[0].set_xlabel('Diagnosis class')

    if 'age' in meta_df.columns:
        meta_df['age'].plot(kind='hist', bins=20, ax=axes[1], color='coral')
        axes[1].set_title('Age Distribution')
        axes[1].set_xlabel('Age')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_html_report(quality_df, dataset_stats, duplicate_report, output_path,
                          class_dist_image_filename='class_distribution.png'):
    corrupted_count = int(quality_df['corrupted'].sum())
    total_images = len(quality_df)

    dims = quality_df[['width', 'height']].describe().to_html()
    mode_counts = quality_df['mode'].value_counts().to_frame('count').to_html()

    dup_html = ""
    if duplicate_report:
        for d in duplicate_report:
            flag = "CROSSES SPLIT BOUNDARY (potential leakage)" if d['crosses_split_boundary'] else "same split / same lesion group - benign"
            dup_html += f"<h4>Hash: {d['hash']} - {flag}</h4><ul>"
            for e in d['entries']:
                dup_html += f"<li>image_id={e['image_id']}, lesion_id={e['lesion_id']}, split={e['split']}</li>"
            dup_html += "</ul>"
    else:
        dup_html = "<p>No duplicate files detected.</p>"

    class_counts_html = pd.Series(dataset_stats['class_counts']).to_frame('count').to_html()
    missing_html = pd.Series(dataset_stats['missing_values']).to_frame('missing_count').to_html()
    sex_html = pd.Series(dataset_stats['sex_counts']).to_frame('count').to_html()
    loc_html = pd.Series(dataset_stats['localization_counts']).to_frame('count').to_html()

    html = f"""
    <html>
    <head>
        <title>DermaAI - Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
            h1 {{ color: #1a5276; }}
            h2 {{ color: #2874a6; border-bottom: 2px solid #ddd; padding-bottom: 4px; margin-top: 40px; }}
            table {{ border-collapse: collapse; margin: 10px 0; }}
            table, th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
            .flag-bad {{ color: #b03a2e; font-weight: bold; }}
            .flag-good {{ color: #1e8449; font-weight: bold; }}
            .summary-box {{ background: #f4f6f7; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>DermaAI - Data Quality Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-box">
            <b>Total images checked:</b> {total_images}<br>
            <b>Corrupted images:</b> <span class="{'flag-bad' if corrupted_count else 'flag-good'}">{corrupted_count}</span><br>
            <b>Class imbalance ratio (majority/minority):</b> {dataset_stats['imbalance_ratio']:.1f}x
        </div>

        <h2>Image Dimensions</h2>
        {dims}

        <h2>Color Mode Distribution</h2>
        {mode_counts}

        <h2>Duplicate File Detection</h2>
        {dup_html}

        <h2>Class Distribution</h2>
        {class_counts_html}
        <img src="{class_dist_image_filename}" width="700">

        <h2>Missing Metadata</h2>
        {missing_html}

        <h2>Sex Distribution</h2>
        {sex_html}

        <h2>Anatomical Site (Localization) Distribution</h2>
        {loc_html}

    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)


def run_data_quality_checks(meta_csv_path, output_dir, path_col='image_path'):
    """Main entry point: runs all checks and writes all report artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    meta_df = pd.read_csv(meta_csv_path)
    image_paths = meta_df[path_col].tolist()

    print("Running image-level checks...")
    quality_df = check_image_quality(image_paths)

    print("Finding duplicates...")
    duplicate_report = find_duplicate_groups(quality_df, meta_df, path_col=path_col)

    print("Running dataset-level analysis...")
    dataset_stats = analyze_dataset_level(meta_df)

    print("Saving metadata report CSV...")
    quality_df.to_csv(os.path.join(output_dir, 'metadata_report.csv'), index=False)

    print("Plotting class distribution...")
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plot_class_distribution(meta_df, class_dist_path)

    print("Generating HTML report...")
    html_path = os.path.join(output_dir, 'data_quality_report.html')
    generate_html_report(quality_df, dataset_stats, duplicate_report, html_path)

    print(f"\nDone. Reports saved to: {output_dir}")
    return {
        'quality_df': quality_df,
        'dataset_stats': dataset_stats,
        'duplicate_report': duplicate_report
    }
