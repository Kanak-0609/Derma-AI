"""backbones.py - torchvision backbone factory and a probability-averaging ensemble."""
import torch, torch.nn as nn
import torchvision.models as tvm


def build_model(name, num_classes=7, pretrained=True):
    w = 'DEFAULT' if pretrained else None
    if name.startswith('efficientnet_b'):
        m = getattr(tvm, name)(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == 'convnext_tiny':
        m = tvm.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == 'resnet50':
        m = tvm.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f'unknown backbone {name}')
    return m


class EnsembleLogProb(nn.Module):
    """Averages member probabilities and returns log(mean p), so softmax(out / T) equals temperature scaling."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        p = torch.stack([torch.softmax(m(x), 1) for m in self.members]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def load_member(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck['model_name'], 7, pretrained=False)
    m.load_state_dict(ck['model_state_dict'])
    return m.to(device).eval()
