"""Fast reference-model loading for parity scripts.

The old pattern (load every shard into a CPU dict -> load_state_dict -> .cuda())
moves the weights three times and peaks at ~2x the checkpoint in host RAM; on a
9 GB checkpoint that is minutes of pure memcpy. `load_model_fast` instead:

  1. preferred: from_pretrained(device_map) — transformers streams safetensors
     straight onto the GPU with mmap, one copy, bounded host RAM;
  2. fallback: shard-at-a-time streaming — open one safetensors shard, copy_
     its tensors into the already-on-GPU model, free it, next shard. One copy
     per tensor, host RAM bounded by a single shard.

Works for any trust_remote_code module mounted under sys.modules (the
VideoChat3 trick) or a normal transformers model class.
"""
import json
import os

import torch


def load_model_fast(ckpt, model_cls, cfg=None, dtype=torch.bfloat16, verbose=True):
    def log(msg):
        if verbose:
            print("[ref_load]", msg, flush=True)

    # --- preferred: transformers streaming path ------------------------------
    try:
        kw = {"dtype": dtype, "device_map": {"": 0}, "low_cpu_mem_usage": True}
        if cfg is not None:
            kw["config"] = cfg
        model = model_cls.from_pretrained(ckpt, **kw)
        return model.eval()
    except Exception as e:
        log(f"from_pretrained fast path unavailable ({type(e).__name__}: {e}); "
            f"falling back to shard streaming")

    # --- fallback: copy_ one shard at a time into the GPU-resident model -----
    if cfg is None:
        raise RuntimeError("shard streaming needs an explicit config")
    model = model_cls._from_config(cfg, dtype=dtype).cuda().eval()

    from safetensors import safe_open

    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())

    weight_map = {}
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {n: "model.safetensors"
                      for n in list(params) + list(bufs) if not n.startswith("lm_head.")}

    by_shard = {}
    for name, fn in weight_map.items():
        by_shard.setdefault(fn, []).append(name)

    loaded = 0
    for fn, names in sorted(by_shard.items()):
        with safe_open(os.path.join(ckpt, fn), framework="pt") as f:
            for name in names:
                tgt = params.get(name)
                if tgt is None:
                    tgt = bufs.get(name)
                if tgt is None or name.startswith("lm_head."):
                    continue   # lm_head is tied (skip, same as the old script)
                tgt.copy_(f.get_tensor(name), non_blocking=True)
                loaded += 1
        torch.cuda.synchronize()
        log(f"shard {fn}: {len(names)} names ({loaded} tensors so far)")
    return model
