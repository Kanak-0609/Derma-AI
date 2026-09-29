"""
train_cv.py - leak-free k-fold training (Phase A, step 2).

Per fold it saves: the final-epoch checkpoint, the training history, and out-of-fold
probabilities. Nothing is selected using the validation fold (fixed epochs, final epoch,
cosine LR). The OOF csv is written last, so its existence marks a finished fold.
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/kaggle/working/repo/src/classification')
from dataset import ClassificationDataset, CLASS_NAMES
from transforms import get_train_transform, get_val_transform
from models import get_model
from train_classification import build_criterion, run_epoch, seed_everything, _worker_init_fn


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for images, _ in loader:
        out.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_fold(fold, split_dir, out_dir, ckpt_dir, model_name='efficientnet_b0', loss_type='weighted_ce',
               image_size=224, batch_size=32, num_epochs=15, lr=1e-4, num_workers=2, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f'{model_name}_{loss_type}_fold{fold}'
    oof_path = f'{out_dir}/oof_{tag}.csv'
    if os.path.exists(oof_path):
        print(f'[skip] {tag} already finished')
        return

    train_csv = f'{split_dir}/fold{fold}_train.csv'
    val_csv = f'{split_dir}/fold{fold}_val.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    print(f'=== {tag} | device={device} | epochs={num_epochs} (final epoch is used) ===')

    train_ds = ClassificationDataset(train_csv, image_size=image_size, transform=get_train_transform(image_size))
    val_ds = ClassificationDataset(val_csv, image_size=image_size, transform=get_val_transform(image_size))
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, generator=g, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = get_model(model_name, num_classes=len(CLASS_NAMES)).to(device)
    criterion = build_criterion(loss_type, train_csv, CLASS_NAMES, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        va_loss, va_m = run_epoch(model, val_loader, criterion, device)   # monitoring only
        scheduler.step()
        print(f'Epoch {epoch}/{num_epochs} | train_loss={tr_loss:.4f} train_f1={tr_m["macro_f1"]:.4f} | '
              f'val_loss={va_loss:.4f} val_acc={va_m["accuracy"]:.4f} val_f1={va_m["macro_f1"]:.4f} '
              f'val_auc={va_m["macro_auc"]:.4f} | {time.time() - t0:.0f}s')
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_macro_f1': tr_m['macro_f1'],
                        'val_loss': va_loss, 'val_accuracy': va_m['accuracy'],
                        'val_macro_f1': va_m['macro_f1'], 'val_macro_auc': va_m['macro_auc']})

    torch.save({'fold': fold, 'model_name': model_name, 'loss_type': loss_type, 'epochs': num_epochs,
                'model_state_dict': model.state_dict()}, f'{ckpt_dir}/{tag}_last.pth')
    pd.DataFrame(history).to_csv(f'{out_dir}/history_{tag}.csv', index=False)

    probs = predict_probs(model, val_loader, device)
    oof = pd.read_csv(val_csv)[['image_id', 'lesion_id', 'diagnosis']].copy()
    assert len(oof) == len(probs), 'row count mismatch between val csv and predictions'
    oof['fold'] = fold
    for i, c in enumerate(CLASS_NAMES):
        oof[f'p_{c}'] = probs[:, i]
    oof.to_csv(oof_path, index=False)          # written last = fold finished
    print(f'Saved {oof_path}')
