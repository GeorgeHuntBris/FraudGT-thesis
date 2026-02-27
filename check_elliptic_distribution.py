import torch, os

data = torch.load(os.path.expanduser('~/FraudGT-thesis/data/Elliptic/processed/data.pt'), weights_only=False)

for split in ['train', 'val', 'test']:
    d = data[split]
    y = d['node'].y
    total = len(y)
    illicit = (y == 1).sum().item()
    licit = (y == 0).sum().item()
    unknown = (y == -1).sum().item()
    labeled = illicit + licit
    ratio = illicit / labeled if labeled > 0 else 0
    print(f"{split}: total={total}, illicit={illicit}, licit={licit}, unknown={unknown}, illicit_ratio={ratio:.4f}")
