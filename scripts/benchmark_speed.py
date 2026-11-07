"""Compare single-kit synthesis latency and saved-phase metrics on the same demos.

Examples:
  python scripts/benchmark_speed.py --device mps
  python scripts/benchmark_speed.py --device cpu
Timers exclude model loading, evaluation and file I/O. The first-case warm-up
is included among 12 times; the median limits its influence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from doe.baselines import gs
from doe.infer import NeuralDOE
from doe.io import phase_to_u8
from doe.metrics import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    ap.add_argument('--checkpoint', default='out/jury/checkpoints/jury_pilot/best.pt')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    demos = sorted(d for d in (ROOT / 'data/demo').iterdir() if d.is_dir())
    cases = [(d.name, np.stack([np.asarray(Image.open(d / f'target_{i}.png')) for i in range(1, 5)])) for d in demos]
    modes = {'network_original': NeuralDOE.load(device=args.device)}
    checkpoint = ROOT / args.checkpoint
    if checkpoint.exists():
        modes['network_finetuned'] = NeuralDOE.load(checkpoint, device=args.device)
    modes['gs_200'] = None
    result = {'device': args.device, 'cases': len(cases), 'modes': {}}
    for name, net in modes.items():
        rows = []
        for case_name, targets in cases:
            start = time.perf_counter()
            phase = gs(targets, iters=200, device=args.device) if net is None else net.predict(targets)
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics = evaluate(phase_to_u8(phase), targets)['summary']
            rows.append({'case': case_name, 'ms': elapsed_ms, 'pcc': metrics['pcc']['mean'],
                         'ssim_global': metrics['ssim_global']['mean']})
        result['modes'][name] = {'median_ms': float(np.median([r['ms'] for r in rows])),
                                 'mean_pcc': float(np.mean([r['pcc'] for r in rows])),
                                 'mean_ssim_global': float(np.mean([r['ssim_global'] for r in rows])),
                                 'rows': rows}
        print(name, 'median_ms', round(result['modes'][name]['median_ms'], 1), flush=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
