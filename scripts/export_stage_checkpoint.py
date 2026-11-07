"""Export an inference-compatible checkpoint with a prefix of learned stages."""
import argparse
import dataclasses
from pathlib import Path

import torch

from doe.distill import truncate_stages
from doe.infer import NeuralDOE
from doe.model import count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--checkpoint", help="source checkpoint; default checkpoints/best.pt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    net = NeuralDOE.load(path=args.checkpoint, device="cpu")
    truncate_stages(net.model, args.stages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.model.state_dict(), "model_cfg": net.model.cfg,
                "loss_cfg": dataclasses.asdict(net.loss_cfg),
                "compression": {"stages": args.stages, "parameters": count_params(net.model),
                                "source": str(net.meta.get("path", ""))}}, args.out)
    print(args.out, count_params(net.model))


if __name__ == "__main__":
    import torch
    main()
