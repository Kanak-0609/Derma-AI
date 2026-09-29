"""train_cv2.py - 5-fold trainer with selectable backbone/resolution and mixed precision.
Same protocol as train_cv.py: fixed epochs, final epoch used, cosine LR, nothing selected on the validation fold."""
import os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

ROOT = '/kaggle/working/repo'
sys.path.insert(0, f'{ROOT}/src/classification')
from dataset import ClassificationDataset, CLASS_NAMES
from transforms import get_train_transform, get_val_transform
from backbones import build_model
from train_classification import build_criterion, seed_everything, _worker_init_fn


def _scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _pass(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, n, P, Y = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot += loss.item() * x.size(0)
            n += x.size(0)
            P.append(torch.softmax(logits.float(), 1).detach().cpu().numpy())
            Y.append(y.cpu().numpy())
    return tot / n, np.concatenate(P), np.concatenate(Y)


def train_fold2(fold, model_name, image_size, split_dir, out_dir, ckpt_dir, batch_size=32, lr=1e-4,
                num_epochs=15, num_workers=2, seed=0, loss_type='weighted_ce'):
    tag = f'{model_name}_r{image_size}_{loss_type}_fold{fold}'
    oof_path = f'{out_dir}/oof_{tag}.csv'
    if os.path.exists(oof_path):
        print(f'[skip] {tag}')
        return
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_csv, val_csv = f'{split_dir}/fold{fold}_train.csv', f'{split_dir}/fold{fold}_val.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    seed_everything(seed)
    print(f'=== {tag} | {device} | amp={use_amp} | epochs={num_epochs} (final epoch used) ===')

    tr = ClassificationDataset(train_csv, image_size=image_size, transform=get_train_transform(image_size))
    va = ClassificationDataset(val_csv, image_size=image_size, transform=get_val_transform(image_size))
    g = torch.Generator()
    g.manual_seed(seed)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                           generator=g, worker_init_fn=_worker_init_fn)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(model_name, len(CLASS_NAMES), pretrained=True).to(device)
    criterion = build_criterion(loss_type, train_csv, CLASS_NAMES, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    scaler = _scaler(use_amp)

    hist = []
    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tl, Ptr, Ytr = _pass(model, tr_loader, criterion, device, opt, scaler, use_amp)
        vl, Pva, Yva = _pass(model, va_loader, criterion, device, use_amp=use_amp)   # monitoring only
        sched.step()
        vf1 = f1_score(Yva, Pva.argmax(1), average='macro', zero_division=0)
        print(f'Epoch {ep}/{num_epochs} | train_loss={tl:.4f} | val_loss={vl:.4f} '
              f'val_acc={(Pva.argmax(1) == Yva).mean():.4f} val_f1={vf1:.4f} | {time.time() - t0:.0f}s')
        hist.append({'epoch': ep, 'train_loss': tl, 'val_loss': vl, 'val_macro_f1': vf1})

    torch.save({'fold': fold, 'model_name': model_name, 'image_size': image_size, 'loss_type': loss_type,
                'epochs': num_epochs, 'model_state_dict': model.state_dict()}, f'{ckpt_dir}/{tag}_last.pth')
    pd.DataFrame(hist).to_csv(f'{out_dir}/history_{tag}.csv', index=False)

    oof = pd.read_csv(val_csv)[['image_id', 'lesion_id', 'diagnosis']].copy()
    assert len(oof) == len(Pva), 'row count mismatch'
    oof['fold'] = fold
    for i, c in enumerate(CLASS_NAMES):
        oof[f'p_{c}'] = Pva[:, i]          # probabilities from the final epoch
    oof.to_csv(oof_path, index=False)      # written last = fold finished
    print(f'Saved {oof_path}')
