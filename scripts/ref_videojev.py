"""Oracle for the VideoJev backend: run the official model and print its scores.

Needs a transformers build that ships ``qwen3_5`` (the reference pins 5.14.1),
so run it in its own environment, e.g.::

  py -3 -m venv .venv-vj
  .venv-vj/Scripts/pip install torch transformers==5.14.1 flash-linear-attention==0.4.2 \
      causal-conv1d av numpy
  .venv-vj/Scripts/python scripts/ref_videojev.py --video clip.mp4 \
      --query "when does the person sit down" --choice "A. inside|B. outside"

The engine side is the same call with ``vqa-cli --score``; compare
``probabilities`` / ``interval`` field by field. Use ``--dump-ids`` to also
print the token ids the reference builds (they must match the engine's prompt
byte for byte, otherwise mrope positions drift).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="models/VideoJev-4B")
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--choice", default="", help="pipe-separated options; empty = grounding")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--dump-ids", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={"": "cuda:0"},
                                      attn_implementation="flash_attention_2")
    model.prepare_for_inference()

    task = {"id": "q0", "task": "grounding" if not args.choice else "choice", "query": args.query}
    if args.choice:
        task["options"] = args.choice.split("|")

    if args.dump_ids:
        # the checkpoint ships its prompt builder next to the weights
        sys.path.insert(0, os.path.abspath(args.model))
        from processing_videojev import prepare_query_groups, prepare_video
        pixels, record = prepare_video(processor, args.video, fps=args.fps,
                                       max_frames=args.max_frames, context=262144)
        for batch, rows in prepare_query_groups(processor, record, [task], 262144):
            print(json.dumps({"grid_thw": record["grid_thw"],
                              "score_times": record["score_times"],
                              "n_tokens": int(batch["input_ids"].shape[1]),
                              "input_ids": batch["input_ids"][0].tolist()}))
        return 0

    results = model.predict_video(processor, args.video, [task], fps=args.fps,
                                  max_frames=args.max_frames, threshold=args.threshold)
    for r in results:
        print(json.dumps({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                          for k, v in r.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
