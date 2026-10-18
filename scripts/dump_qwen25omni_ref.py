"""Dump Qwen2.5-Omni reference tensors from the official transformers stack.

Everything the C++ backend needs for parity debugging, one directory:

  meta.json            token ids, shapes, generation params, text answer
  mel.npy              log-mel exactly as the HF feature extractor made it
  audio_feats.npy      audio tower output [N_audio, 2048]
  vision_feats.npy     vision tower output [N_img, 2048]
  positions.npy        thinker MRoPE positions [3, T_prompt] (get_rope_index)
  logits0/1/2.npy      thinker logits for the first three decode steps
  gen_tokens.npy       thinker generated token ids
  wav.npy              token2wav output at 24 kHz (float32, [-1, 1])

Usage::

  python scripts/dump_qwen25omni_ref.py --src models/qwen25omni/hf \
      -o models/qwen25omni/ref --image test.jpg --audio test.wav \
      --prompt "..." [--max-new 24]
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--prompt", default="Describe this in detail.")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--speaker", default="Chelsie")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="float32 avoids the talker speaker-conditioning dtype clash "
                         "(cond is force-cast .float() in HF, matmul fails under bf16)")
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    os.makedirs(args.out, exist_ok=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.src, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        device_map="cpu", low_cpu_mem_usage=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.src)

    # ---------------------------------------------------------------- inputs
    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    content.append({"type": "text", "text": args.prompt})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                         tokenize=False)
    audios, images = None, None
    if args.audio:
        import librosa
        audios = [librosa.load(args.audio, sr=16000, mono=True)[0]]
    if args.image:
        from PIL import Image
        images = [Image.open(args.image).convert("RGB")]
    inputs = processor(text=text, audio=audios, images=images,
                       return_tensors="pt", padding=True)

    meta = {"prompt_text": text, "input_ids": inputs["input_ids"][0].tolist()}
    print("prompt tokens:", len(meta["input_ids"]))

    if "input_features" in inputs:
        mel = inputs["input_features"][0].float()   # [n_mel, T_pad]
        if "feature_attention_mask" in inputs:
            mel = mel[:, :int(inputs["feature_attention_mask"][0].sum())]
        mel = mel.T                                  # -> [T, n_mel] for the dump
        np.save(os.path.join(args.out, "mel.npy"), mel.numpy())
        meta["mel_shape"] = list(mel.shape)
        if "feature_attention_mask" in inputs:
            meta["feature_len"] = int(inputs["feature_attention_mask"][0].sum())

    with torch.no_grad():
        thinker = model.thinker
        feats = {}
        if "pixel_values" in inputs:
            vis = thinker.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            feats["vision_feats"] = vis.last_hidden_state if hasattr(vis, "last_hidden_state") else vis
        if "input_features" in inputs:
            flens = (inputs["feature_attention_mask"].sum(-1)
                     if "feature_attention_mask" in inputs else None)
            feat = inputs["input_features"][0]     # [n_mel, T] unbatched
            if flens is not None:      # trim the processor's 300 s padding
                feat = feat[:, :int(flens[0])]
            aud = thinker.audio_tower(feat, feature_lens=flens)
            feats["audio_feats"] = aud.last_hidden_state if hasattr(aud, "last_hidden_state") else aud[0]
        for k, v in feats.items():
            arr = v.float().cpu().numpy()
            np.save(os.path.join(args.out, f"{k}.npy"), arr)
            meta[k] = list(arr.shape)
            print(k, arr.shape)

        pos, deltas = thinker.get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            None,
            False,
            inputs.get("feature_attention_mask").sum(-1)
            if "feature_attention_mask" in inputs else None,
            None,
        )
        np.save(os.path.join(args.out, "positions.npy"),
                pos[:, 0].cpu().numpy().astype(np.int64))
        meta["mrope_deltas"] = int(deltas[0])

        out = thinker.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:].cpu().numpy()
        np.save(os.path.join(args.out, "gen_tokens.npy"), gen_ids)
        meta["gen_tokens"] = gen_ids.tolist()
        for i in range(min(3, len(out.scores))):
            np.save(os.path.join(args.out, f"logits{i}.npy"),
                    out.scores[i][0].float().cpu().numpy())
        meta["gen_text"] = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # ------------------------------------------------------- talker + t2w
    model.enable_talker()
    try:
        with torch.no_grad():
            text_ids, wav = model.generate(
                input_ids=inputs["input_ids"],
                speaker=args.speaker,
                use_audio_in_video=False,
                thinker_max_new_tokens=args.max_new,
                talker_do_sample=False,
            )
            if wav is not None:
                wav = wav.float().cpu().numpy()
                np.save(os.path.join(args.out, "wav.npy"), wav)
                meta["wav_samples"] = int(wav.shape[-1])
                meta["speaker"] = args.speaker
    except Exception as e:      # talker path is optional for thinker parity
        meta["talker_error"] = str(e)
        print("talker/t2w skipped:", e)

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("done:", args.out)
    print("text:", meta.get("gen_text", "")[:200])


if __name__ == "__main__":
    raise SystemExit(main())
