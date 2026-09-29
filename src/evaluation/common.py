"""common.py - shared helpers (analysis scripts, checkpoint backup/restore)."""
import os, sys, zipfile, subprocess, importlib.util
import numpy as np, torch
from PIL import Image

ROOT = '/kaggle/working/repo'
REPO = 'Kanak-0609/Derma-AI'
CKPT = '/kaggle/working/models_cv'
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL = CLASS_NAMES.index('mel')
sys.path.insert(0, f'{ROOT}/src/classification')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clf_val_transform(size):
    return _load('clf_transforms', f'{ROOT}/src/classification/transforms.py').get_val_transform(size)


def apply_tf(tf, pil):
    arr = np.array(pil)
    try:
        return tf(image=arr)['image']
    except TypeError:
        return tf(pil)


@torch.no_grad()
def predict_paths(model, paths, tf, device, pil_fn=None, batch_size=32):
    model.eval()
    out = []
    for i in range(0, len(paths), batch_size):
        xs = []
        for p in paths[i:i + batch_size]:
            pil = Image.open(p).convert('RGB')
            if pil_fn is not None:
                pil = pil_fn(pil)
            xs.append(apply_tf(tf, pil))
        x = torch.stack(xs).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            lg = model(x)
        out.append(torch.softmax(lg.float(), 1).cpu().numpy())
    return np.concatenate(out)


def push(paths, msg):
    os.chdir(ROOT)
    for cmd in (['git', 'add'] + list(paths), ['git', 'commit', '-m', msg], ['git', 'push']):
        r = subprocess.run(cmd, capture_output=True, text=True)
    print('push ->', (r.stdout + r.stderr).strip()[-120:])


def github_token():
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret('GITHUB_TOKEN')


def upload_release(tag, zip_path):
    import requests
    H = {'Authorization': f'token {github_token()}', 'Accept': 'application/vnd.github+json'}
    api = f'https://api.github.com/repos/{REPO}/releases'
    r = requests.get(f'{api}/tags/{tag}', headers=H)
    if r.status_code == 404:
        r = requests.post(api, headers=H, json={'tag_name': tag, 'name': tag, 'body': 'model files'})
    r.raise_for_status()
    rel = r.json()
    name = os.path.basename(zip_path)
    for a in rel.get('assets', []):
        if a['name'] == name:
            requests.delete(a['url'], headers=H)
    with open(zip_path, 'rb') as f:
        u = requests.post(rel['upload_url'].split('{')[0], params={'name': name}, data=f,
                          headers={**H, 'Content-Type': 'application/zip'})
    u.raise_for_status()
    return u.json()['browser_download_url']


def backup_checkpoints(tag, release='checkpoints-cv'):
    files = [f'{CKPT}/{tag}_fold{k}_last.pth' for k in range(5)]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 5:
        print(f'[backup skipped] only {len(files)}/5 checkpoints for {tag} exist in this session')
        return None
    zp = f'/kaggle/working/{tag}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    return upload_release(release, zp)


def ensure_checkpoints(tag, release='checkpoints-cv'):
    need = [f'{CKPT}/{tag}_fold{k}_last.pth' for k in range(5)]
    if all(os.path.exists(p) for p in need):
        return
    os.makedirs(CKPT, exist_ok=True)
    zp = f'/kaggle/working/{tag}.zip'
    os.system(f'wget -q -O {zp} https://github.com/{REPO}/releases/download/{release}/{tag}.zip')
    with zipfile.ZipFile(zp) as z:
        z.extractall(CKPT)


def ensure_release_models():
    if os.path.exists('/kaggle/working/models_from_release/segmentation/unet_best.pth'):
        return
    zp = '/kaggle/working/all_checkpoints_backup.zip'
    os.system(f'wget -q -O {zp} https://github.com/{REPO}/releases/download/checkpoints-run2/all_checkpoints_backup.zip')
    with zipfile.ZipFile(zp) as z:
        z.extractall('/kaggle/working/models_from_release')
