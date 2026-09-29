"""
explain_wrappers.py - mask_fn (U-Net) and cam_fn (Grad-CAM) for DecisionSupport.
"""
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F

SEG_DIR = '/kaggle/working/repo/src/segmentation'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply(transform, pil):
    arr = np.array(pil)
    try:                                   # albumentations style
        return transform(image=arr)['image']
    except TypeError:                      # torchvision style
        return transform(pil)


def make_mask_fn(unet_ckpt, device, image_size=256, threshold=0.5, src_dir=SEG_DIR):
    unet_mod = _load('seg_unet', f'{src_dir}/unet.py')
    tf_mod = _load('seg_transforms', f'{src_dir}/transforms.py')
    net = unet_mod.UNet(in_channels=3, out_channels=1).to(device)
    ck = torch.load(unet_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck['model_state_dict'])
    net.eval()
    tf = tf_mod.get_val_transform(image_size)

    @torch.no_grad()
    def mask_fn(pil):
        x = _apply(tf, pil).unsqueeze(0).to(device)
        out = net(x)[0, 0]
        mask_fn.last_raw = out.cpu().numpy()          # kept for sanity checks
        prob = torch.sigmoid(out).cpu().numpy()
        return (prob >= threshold).astype(np.uint8)   # (image_size, image_size)
    return mask_fn


def make_cam_fn(model, transform, device, target_layer=None):
    layer = target_layer if target_layer is not None else model.features[-1]
    store = {}

    def fwd_hook(module, inp, out):
        store['act'] = out
        if out.requires_grad:
            out.register_hook(lambda g: store.__setitem__('grad', g))
    layer.register_forward_hook(fwd_hook)

    def cam_fn(pil, class_idx):
        x = _apply(transform, pil).unsqueeze(0).to(device)
        with torch.enable_grad():                     # predict() runs under no_grad
            model.zero_grad()
            logits = model(x)
            logits[0, int(class_idx)].backward()
        act = store['act'][0].detach()
        grad = store['grad'][0]
        w = grad.mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * act).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode='bilinear',
                            align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()                      # (H, W) in [0, 1], classifier input size
    return cam_fn
