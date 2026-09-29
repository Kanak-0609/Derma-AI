"""experiments.py - single source of truth for the Phase B comparison."""
BASELINE = dict(tag='efficientnet_b0_weighted_ce', model_name='efficientnet_b0', image_size=224, batch_size=32, lr=1e-4)
CANDIDATES = [
    dict(model_name='efficientnet_b0', image_size=320, batch_size=32, lr=1e-4),
    dict(model_name='convnext_tiny', image_size=224, batch_size=32, lr=5e-5),
    dict(model_name='efficientnet_b3', image_size=300, batch_size=24, lr=1e-4),
]
for _c in CANDIDATES:
    _c['tag'] = f"{_c['model_name']}_r{_c['image_size']}_weighted_ce"
ALL = [BASELINE] + CANDIDATES
