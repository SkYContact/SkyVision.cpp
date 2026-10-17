"""Reference-side scoring for VideoJev parity checks (official code, no engine).

Feeds a directory of frames through the checkpoint's own preprocessing and
decision head, printing the same JSON shape as ``vqa-cli --score`` so the two
can be diffed field by field.

Needs a transformers build that ships qwen3_5 (the checkpoint pins 5.14.1):

    py -3 -m venv .venv-vj
    .venv-vj/Scripts/python.exe -m pip install "transformers==5.14.1" numpy pillow
    # torch can be reused from .venv via a site-packages .pth

  .venv-vj/Scripts/python.exe scripts/parity_videojev.py --frames tmp_frames \
      --fps 2 --questions tmp_qs.json --model models/VideoJev-4B

``--size WxH`` forces the pixel budget so the reference keeps the same grid the
engine chose (the official prepare_video maximizes resolution instead).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="models/VideoJev-4B")
    ap.add_argument("--frames", required=True, help="directory of PNG/JPG frames, sorted by name")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--size", default="", help="WxH pixel budget, e.g. 320x256 (default: native)")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--dump-ids", action="store_true",
                    help="print the official prompt token ids / candidate indices only")
    ap.add_argument("--visual-dtype", default="fp32", choices=["fp32", "float16", "bfloat16"],
                    help="tower compute dtype; the official entry point uplifts it to fp32, "
                         "the engine runs float16 (ablation)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="reference weight dtype (float32 on cpu = ground truth, slow)")
    ap.add_argument("--context", type=int, default=262144)
    args = ap.parse_args()

    # the checkpoint's modules use relative imports, so import them as a package
    # (the directory name itself is not a valid python identifier)
    import importlib
    import shutil
    import tempfile
    pkg_root = Path(tempfile.mkdtemp(prefix="vjref_"))
    pkg = pkg_root / "videojev_ref"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for src in Path(args.model).glob("*.py"):
        shutil.copy(src, pkg / src.name)
    sys.path.insert(0, str(pkg_root))
    _proc = importlib.import_module("videojev_ref.processing_videojev")
    _temp = importlib.import_module("videojev_ref.videojev_temporal")
    format_predictions = _proc.format_predictions
    prepare_query_groups = _proc.prepare_query_groups
    pair_timeline = _temp.pair_timeline
    from transformers import AutoModel, AutoProcessor
    from transformers.video_utils import VideoMetadata

    files = sorted(p for p in Path(args.frames).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    frames = [np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB"))
              for p in files]
    n = len(frames)
    duration = n / args.fps

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    target = "cpu" if args.device == "cpu" else "cuda:0"
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      dtype=getattr(torch, args.dtype),
                                      device_map={"": target},
                                      attn_implementation=args.attn)
    model.prepare_for_inference()
    print(f"# weights {next(model.parameters()).dtype} on {next(model.parameters()).device}",
          file=sys.stderr, flush=True)
    if args.visual_dtype != "fp32":
        # the official entry point uplifts the tower to fp32; the engine keeps it
        # f16, so this ablates that single difference
        model.backbone.visual.to(getattr(torch, args.visual_dtype))

    vp = processor.video_processor
    resize_kwargs = {}
    if args.size:
        # resize here instead: the processor's smart_resize maximizes or clamps by
        # area, so the only way to hit the engine's exact grid is to hand it the
        # final resolution and skip its own resize
        from PIL import Image
        w, h = (int(v) for v in args.size.lower().split("x"))
        frames = [np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in frames]
        resize_kwargs = dict(do_resize=False)
    else:
        size = dict(shortest_edge=vp.size["shortest_edge"], longest_edge=n * 640 * 640)
        resize_kwargs = dict(size=size)
    meta = VideoMetadata(total_num_frames=n, fps=args.fps, frames_indices=list(range(n)))
    native = vp(videos=[np.stack(frames)], video_metadata=[meta], do_sample_frames=False,
                return_tensors="pt", **resize_kwargs)
    grid = native["video_grid_thw"][0].tolist()
    times = np.arange(int(np.ceil(duration * args.fps))) / args.fps
    centers, bounds = pair_timeline(times, duration)
    record = dict(frames=n, fps=args.fps, grid_thw=grid, tokens=int(np.prod(grid)) // 4,
                  score_times=centers.tolist(), score_bounds=bounds.tolist(),
                  sampled_duration=duration, effective_fps=args.fps)

    with open(args.questions, encoding="utf-8") as f:
        rows = json.load(f)["questions"]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i))

    if args.dump_ids:
        batches = []
        for batch, _group in prepare_query_groups(processor, record, rows, args.context):
            batches.append(dict(input_ids=batch["input_ids"][0].tolist(),
                                pair_indices=batch["pair_indices"].tolist(),
                                query_indices=batch["query_indices"].tolist()))
        print(json.dumps({"grid_thw": grid, "score_times": centers.tolist(),
                          "batches": batches}, ensure_ascii=False))
        return 0

    features = model.encode_video(native["pixel_values_videos"], torch.tensor([grid]))
    results, groups = [], []
    for batch, group in prepare_query_groups(processor, record, rows, args.context):
        out = model(**batch, visual_features=features)
        results.extend(format_predictions(out, group, record, dict(model.config.decoder)))
        groups.append(int(batch["input_ids"].shape[1]))
    print(json.dumps({"grid_thw": grid, "score_times": centers.tolist(),
                      "prompt_tokens": groups, "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
