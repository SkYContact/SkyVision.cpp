"""Post-conversion self-check for minicpmo-vocoder.gguf (no model load).

Usage::

  python convert/check_minicpmo45_vocoder.py [--gguf OUT.gguf] [--src DIR]
                                             [--dump DIR]

Verifies, without loading the model:
  1. GGUF tensor name set == the converter plan recomputed from the sources
     (nothing extra, nothing missing - diff sets empty)
  2. per-tensor ne/dtype sanity against the plan (F16 operands / F32 norms)
  3. metadata KV spot checks (dims, block counts, eps values, prompt lens)
  4. round-trip byte-exact recomputation of sampled tensors straight from
     flow.pt / hift.pt / campplus.onnx (weight_norm recombine, ConvTranspose
     transpose, BN folds incl. the transit affines, prompt cache, rand_noise)

Exit code 0 = PASS.
"""
import argparse
import os
import sys

import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BN_EPS = 1e-5
GGML_F16 = gguf.GGMLQuantizationType.F16
GGML_F32 = gguf.GGMLQuantizationType.F32


def kv_dict(reader):
    out = {}
    for k, f in reader.fields.items():
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            out[k] = None
        else:
            c = f.contents
            vals = c() if callable(c) else c
            if isinstance(vals, (list, tuple)):
                vals = vals[0] if len(vals) == 1 else vals
            out[k] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="E:/SkyVision-weights/minicpmo45/minicpmo-vocoder.gguf")
    ap.add_argument("--src", default="E:/SkyVision-weights/minicpmo45/assets/token2wav")
    ap.add_argument("--dump", default="D:/S2TT/minicpmo45-port/4vocoder")
    args = ap.parse_args()

    import convert_minicpmo45_vocoder as C

    # ---- expected plan: rerun the converter against a null writer ----------
    plan = {}
    S = C.Sources(args.src)
    flow, hift = S.flow, S.hift

    def exp(name, arr, dtype):
        plan[name] = (np.ascontiguousarray(arr), dtype)

    def e_lin(name, w_oi, b=None):
        exp(name + ".weight", np.asarray(w_oi), GGML_F16)   # torch [out, in]
        if b is not None:
            plan[name + ".bias"] = (np.asarray(b, np.float32), GGML_F32)

    def e_conv(name, wt, b=None):
        if wt.ndim == 3:
            wt = wt.reshape(wt.shape[0], -1)
        exp(name + ".weight", wt, GGML_F16)
        if b is not None:
            plan[name + ".bias"] = (np.asarray(b, np.float32), GGML_F32)

    def e_conv3d(name, wt, b=None):
        exp(name + ".weight", wt, GGML_F16)   # torch [OC, IC, K] kept 3-D
        if b is not None:
            plan[name + ".bias"] = (np.asarray(b, np.float32), GGML_F32)

    z = np.load(f"{args.dump}/z.npy")[0]
    tail = np.load(f"{args.dump}/rand_noise_full.npy")
    n = z.shape[1]
    exp("t2w.rand_noise", np.concatenate([z, tail[:, n:]], axis=1), GGML_F32)
    plan["t2w.prompt.tokens"] = (np.load(f"{args.dump}/prompt_speech_tokens.npy")
                                 .reshape(-1).astype(np.int32), None)
    pm = np.load(f"{args.dump}/prompt_mels.npy")
    exp("t2w.prompt.mels", pm.reshape(pm.shape[1], 80), GGML_F32)
    fb = np.load(f"{args.dump}/spk_fbank.npy")
    exp("t2w.spk.fbank", fb.reshape(fb.shape[1], 80), GGML_F32)
    exp("t2w.spk.emb_ref",
        np.load(f"{args.dump}/spk_emb_ort.npy").reshape(-1), GGML_F32)

    exp("cfm.input_emb.weight", C.to_np(flow["input_embedding.weight"]), GGML_F16)
    e_lin("cfm.spk_proj", C.to_np(flow["spk_embed_affine_layer.weight"]),
          C.to_np(flow["spk_embed_affine_layer.bias"]))
    e_lin("cfm.embed", C.to_np(flow["encoder.embed.out.0.weight"]),
          C.to_np(flow["encoder.embed.out.0.bias"]))
    plan["cfm.embed_ln.weight"] = (C.to_np(flow["encoder.embed.out.1.weight"]), GGML_F32)
    plan["cfm.embed_ln.bias"] = (C.to_np(flow["encoder.embed.out.1.bias"]), GGML_F32)
    e_conv3d("cfm.lookahead.conv1", C.to_np(flow["encoder.pre_lookahead_layer.conv1.weight"]),
             C.to_np(flow["encoder.pre_lookahead_layer.conv1.bias"]))
    e_conv3d("cfm.lookahead.conv2", C.to_np(flow["encoder.pre_lookahead_layer.conv2.weight"]),
             C.to_np(flow["encoder.pre_lookahead_layer.conv2.bias"]))

    def blk_expect(mod, i, gname):
        b = f"cfm.{gname}{i}"
        for a, h in (("ln_mha", "norm_mha"), ("ln_ff", "norm_ff")):
            plan[f"{b}.{a}.weight"] = (C.to_np(flow[f"encoder.{mod}.{i}.{h}.weight"]), GGML_F32)
            plan[f"{b}.{a}.bias"] = (C.to_np(flow[f"encoder.{mod}.{i}.{h}.bias"]), GGML_F32)
        base = f"encoder.{mod}.{i}.self_attn"
        for nm, h in (("wq", "linear_q"), ("wk", "linear_k"),
                      ("wv", "linear_v"), ("wo", "linear_out"),
                      ("wpos", "linear_pos")):
            bias = None if h == "linear_pos" else C.to_np(flow[f"{base}.{h}.bias"])
            e_lin(f"{b}.{nm}", C.to_np(flow[f"{base}.{h}.weight"]), bias)
        plan[f"{b}.pos_bias_u"] = (C.to_np(flow[f"{base}.pos_bias_u"]).reshape(-1), GGML_F32)
        plan[f"{b}.pos_bias_v"] = (C.to_np(flow[f"{base}.pos_bias_v"]).reshape(-1), GGML_F32)
        e_lin(f"{b}.ffn_gate", C.to_np(flow[f"encoder.{mod}.{i}.feed_forward.w_1.weight"]),
              C.to_np(flow[f"encoder.{mod}.{i}.feed_forward.w_1.bias"]))
        e_lin(f"{b}.ffn_down", C.to_np(flow[f"encoder.{mod}.{i}.feed_forward.w_2.weight"]),
              C.to_np(flow[f"encoder.{mod}.{i}.feed_forward.w_2.bias"]))

    for i in range(6):
        blk_expect("encoders", i, "blk.")
    e_conv3d("cfm.up.conv", C.to_np(flow["encoder.up_layer.conv.weight"]),
             C.to_np(flow["encoder.up_layer.conv.bias"]))
    e_lin("cfm.up_embed", C.to_np(flow["encoder.up_embed.out.0.weight"]),
          C.to_np(flow["encoder.up_embed.out.0.bias"]))
    plan["cfm.up_embed_ln.weight"] = (C.to_np(flow["encoder.up_embed.out.1.weight"]), GGML_F32)
    plan["cfm.up_embed_ln.bias"] = (C.to_np(flow["encoder.up_embed.out.1.bias"]), GGML_F32)
    for i in range(4):
        blk_expect("up_encoders", i, "ublkc.")
    plan["cfm.after_norm.weight"] = (C.to_np(flow["encoder.after_norm.weight"]), GGML_F32)
    plan["cfm.after_norm.bias"] = (C.to_np(flow["encoder.after_norm.bias"]), GGML_F32)
    e_lin("cfm.proj", C.to_np(flow["encoder_proj.weight"]),
          C.to_np(flow["encoder_proj.bias"]))

    est = "decoder.estimator"
    e_lin("dit.t_mlp0", C.to_np(flow[f"{est}.t_embedder.mlp.0.weight"]),
          C.to_np(flow[f"{est}.t_embedder.mlp.0.bias"]))
    e_lin("dit.t_mlp1", C.to_np(flow[f"{est}.t_embedder.mlp.2.weight"]),
          C.to_np(flow[f"{est}.t_embedder.mlp.2.bias"]))
    e_lin("dit.in_proj", C.to_np(flow[f"{est}.in_proj.weight"]),
          C.to_np(flow[f"{est}.in_proj.bias"]))
    for i in range(16):
        b, p = f"dit.blk.{i}", f"{est}.blocks.{i}"
        for nm, h in (("wq", "to_q"), ("wk", "to_k"), ("wv", "to_v"),
                      ("wo", "proj")):
            e_lin(f"{b}.{nm}", C.to_np(flow[f"{p}.attn.{h}.weight"]),
                  C.to_np(flow[f"{p}.attn.{h}.bias"]))
        for nm in ("q_norm", "k_norm"):
            plan[f"{b}.{nm}.weight"] = (C.to_np(flow[f"{p}.attn.{nm}.weight"]), GGML_F32)
            plan[f"{b}.{nm}.bias"] = (C.to_np(flow[f"{p}.attn.{nm}.bias"]), GGML_F32)
        e_lin(f"{b}.mlp_gate", C.to_np(flow[f"{p}.mlp.fc1.weight"]),
              C.to_np(flow[f"{p}.mlp.fc1.bias"]))
        e_lin(f"{b}.mlp_down", C.to_np(flow[f"{p}.mlp.fc2.weight"]),
              C.to_np(flow[f"{p}.mlp.fc2.bias"]))
        e_conv(f"{b}.conv1", C.to_np(flow[f"{p}.conv.block.1.weight"]),
               C.to_np(flow[f"{p}.conv.block.1.bias"]))
        plan[f"{b}.conv_ln.weight"] = (C.to_np(flow[f"{p}.conv.block.3.weight"]), GGML_F32)
        plan[f"{b}.conv_ln.bias"] = (C.to_np(flow[f"{p}.conv.block.3.bias"]), GGML_F32)
        e_conv(f"{b}.conv2", C.to_np(flow[f"{p}.conv.block.6.weight"]),
               C.to_np(flow[f"{p}.conv.block.6.bias"]))
        e_lin(f"{b}.adaln", C.to_np(flow[f"{p}.adaLN_modulation.1.weight"]),
              C.to_np(flow[f"{p}.adaLN_modulation.1.bias"]))
    e_lin("dit.fin.adaln", C.to_np(flow[f"{est}.final_layer.adaLN_modulation.1.weight"]),
          C.to_np(flow[f"{est}.final_layer.adaLN_modulation.1.bias"]))
    e_lin("dit.fin.out", C.to_np(flow[f"{est}.final_layer.linear.weight"]),
          C.to_np(flow[f"{est}.final_layer.linear.bias"]))

    for i, ci in enumerate((0, 2, 4, 6, 8)):
        e_conv(f"hifi.f0.c{i}", S.wn(f"f0_predictor.condnet.{ci}"),
               C.to_np(hift[f"f0_predictor.condnet.{ci}.bias"]))
    e_lin("hifi.f0.head", C.to_np(hift["f0_predictor.classifier.weight"]),
          C.to_np(hift["f0_predictor.classifier.bias"]))
    plan["hifi.src_linear.weight"] = (C.to_np(hift["m_source.l_linear.weight"]), GGML_F16)
    plan["hifi.src_linear.bias"] = (C.to_np(hift["m_source.l_linear.bias"]), GGML_F32)
    e_conv3d("hifi.conv_pre", S.wn("conv_pre"), C.to_np(hift["conv_pre.bias"]))
    for i in range(3):
        e_conv(f"hifi.up.{i}", S.wn(f"ups.{i}"),
               C.to_np(hift[f"ups.{i}.bias"]))
    ri = 0
    for stage in range(3):
        for j in range(3):
            for c in (1, 2):
                for k in range(3):
                    e_conv(f"hifi.rb.{ri}.c{c}.{k}",
                           S.wn(f"resblocks.{ri}.convs{c}.{k}"),
                           C.to_np(hift[f"resblocks.{ri}.convs{c}.{k}.bias"]))
            for k in range(3):
                for half in (1, 2):
                    plan[f"hifi.rb.{ri}.a{half}.{k}"] = (
                        C.to_np(hift[f"resblocks.{ri}.activations{half}.{k}.alpha"]),
                        GGML_F32)
            ri += 1
    for i in range(3):
        e_conv(f"hifi.sdown.{i}", C.to_np(hift[f"source_downs.{i}.weight"]),
               C.to_np(hift[f"source_downs.{i}.bias"]))
    for j in range(3):
        for c in (1, 2):
            for k in range(3):
                e_conv(f"hifi.srb.{j}.c{c}.{k}",
                       S.wn(f"source_resblocks.{j}.convs{c}.{k}"),
                       C.to_np(hift[f"source_resblocks.{j}.convs{c}.{k}.bias"]))
        for k in range(3):
            for half in (1, 2):
                plan[f"hifi.srb.{j}.a{half}.{k}"] = (
                    C.to_np(hift[f"source_resblocks.{j}.activations{half}.{k}.alpha"]),
                    GGML_F32)
    e_conv3d("hifi.conv_post", S.wn("conv_post"), C.to_np(hift["conv_post.bias"]))

    # campplus (mirror of the converter stores; all xvector BNs standalone)
    def e_bn(prefix, name):
        g, beta, mean, var = S.bn(name)
        plan[prefix + ".bn.w"] = (g, GGML_F32)
        plan[prefix + ".bn.b"] = (beta, GGML_F32)
        plan[prefix + ".bn.m"] = (mean, GGML_F32)
        plan[prefix + ".bn.v"] = (var, GGML_F32)
    for gname, node in (("spk.head.conv1", "/head/conv1/Conv"),
                        ("spk.layer1.0.conv1", "/head/layer1/layer1.0/conv1/Conv"),
                        ("spk.layer1.0.conv2", "/head/layer1/layer1.0/conv2/Conv"),
                        ("spk.layer1.0.shortcut", "/head/layer1/layer1.0/shortcut/shortcut.0/Conv"),
                        ("spk.layer1.1.conv1", "/head/layer1/layer1.1/conv1/Conv"),
                        ("spk.layer1.1.conv2", "/head/layer1/layer1.1/conv2/Conv"),
                        ("spk.layer2.0.conv1", "/head/layer2/layer2.0/conv1/Conv"),
                        ("spk.layer2.0.conv2", "/head/layer2/layer2.0/conv2/Conv"),
                        ("spk.layer2.0.shortcut", "/head/layer2/layer2.0/shortcut/shortcut.0/Conv"),
                        ("spk.layer2.1.conv1", "/head/layer2/layer2.1/conv1/Conv"),
                        ("spk.layer2.1.conv2", "/head/layer2/layer2.1/conv2/Conv"),
                        ("spk.conv2", "/head/conv2/Conv")):
        wt, bt = S.cp_conv(node)
        e_conv(gname, wt, bt)
    wt, bt = S.cp_conv("/xvector/tdnn/linear/Conv")
    e_conv3d("spk.tdnn", wt, bt)
    e_bn("spk.blk1.tdnnd1", "xvector.block1.tdnnd1.nonlinear1.batchnorm")
    e_bn("spk.blk2.tdnnd1", "xvector.block2.tdnnd1.nonlinear1.batchnorm")
    e_bn("spk.blk3.tdnnd1", "xvector.block3.tdnnd1.nonlinear1.batchnorm")
    NB = {1: 12, 2: 24, 3: 16}
    for bi in (1, 2, 3):
        for ti in range(1, NB[bi] + 1):
            p = f"/xvector/block{bi}/tdnnd{ti}"
            b = f"spk.blk{bi}.tdnnd{ti}"
            if ti >= 2:
                e_bn(b, f"xvector.block{bi}.tdnnd{ti}.nonlinear1.batchnorm")
            wname, bname = S.convs[f"{p}/linear1/Conv"]
            e_lin(f"{b}.linear1", S.cp_t(wname).squeeze(-1), S.cp_t(bname))
            wname, _ = S.convs[f"{p}/cam_layer/linear_local/Conv"]
            e_conv3d(f"{b}.cam_ll", S.cp_t(wname))
            wname, bname = S.convs[f"{p}/cam_layer/linear1/Conv"]
            e_lin(f"{b}.cam1", S.cp_t(wname).squeeze(-1), S.cp_t(bname))
            wname, bname = S.convs[f"{p}/cam_layer/linear2/Conv"]
            e_lin(f"{b}.cam2", S.cp_t(wname).squeeze(-1), S.cp_t(bname))
    transit_nodes = {n.name.split("/")[2]: n for n in S.onnx_model.graph.node
                     if n.op_type == "Conv" and "/transit" in n.name
                     and n.name.endswith("linear/Conv")}
    for ti in (1, 2, 3):
        wt, bt = S.cp_conv(transit_nodes[f"transit{ti}"].name)
        e_lin(f"spk.transit{ti}.conv", wt.reshape(wt.shape[0], -1), bt)
        e_bn(f"spk.transit{ti}", f"xvector.transit{ti}.nonlinear.batchnorm")
    wname, _ = S.convs["/xvector/dense/linear/Conv"]
    mean = S.cp_t("xvector.dense.nonlinear.batchnorm.running_mean")
    var = S.cp_t("xvector.dense.nonlinear.batchnorm.running_var")
    sc = 1.0 / np.sqrt(var + BN_EPS)
    plan["spk.dense.weight"] = ((S.cp_t(wname).reshape(192, 1024) * sc.reshape(-1, 1)), GGML_F16)
    plan["spk.dense.bias"] = (-mean * sc, GGML_F32)

    # ---- read the GGUF ------------------------------------------------------
    reader = gguf.GGUFReader(args.gguf)
    got = {t.name: t for t in reader.tensors}
    missing = sorted(set(plan) - set(got))
    extra = sorted(set(got) - set(plan))
    if missing:
        C.fail(f"missing {len(missing)} tensors: {missing[:10]}")
    if extra:
        C.fail(f"extra {len(extra)} tensors: {extra[:10]}")

    bad = 0
    for name, (arr, dtype) in plan.items():
        t = got[name]
        want_type = dtype if dtype is not None else GGML_F32
        if name == "t2w.prompt.tokens":
            want_type = None        # I32
            if int(t.tensor_type) != 26 or list(t.shape) != [arr.size]:
                print(f"BAD {name}: {t.tensor_type} {t.shape}")
                bad += 1
            continue
        if t.tensor_type != want_type:
            print(f"BAD {name}: dtype {t.tensor_type} != {want_type}")
            bad += 1
        want_ne = list(arr.shape[::-1])
        if list(t.shape) != want_ne:
            print(f"BAD {name}: ne {t.shape} != {want_ne}")
            bad += 1
    if bad:
        C.fail(f"{bad} tensors with wrong dtype/shape")

    # ---- byte-exact sampling ------------------------------------------------
    SAMPLE = [
        "t2w.rand_noise", "t2w.prompt.mels", "t2w.spk.fbank", "t2w.spk.emb_ref",
        "cfm.input_emb.weight", "cfm.blk.0.wq.weight", "cfm.blk.5.wo.bias",
        "cfm.up.conv.weight", "cfm.ublkc.3.ffn_down.weight", "cfm.proj.weight",
        "dit.blk.0.adaln.weight", "dit.blk.15.k_norm.bias", "dit.fin.out.weight",
        "hifi.f0.c0.weight", "hifi.conv_pre.weight", "hifi.up.0.weight",
        "hifi.rb.4.c1.2.weight", "hifi.rb.8.a2.1", "hifi.sdown.0.weight",
        "hifi.conv_post.bias",
        "spk.head.conv1.weight", "spk.tdnn.weight", "spk.tdnn.bias",
        "spk.blk1.tdnnd2.bn.w", "spk.blk1.tdnnd2.bn.v",
        "spk.blk2.tdnnd1.bn.m", "spk.blk2.tdnnd13.linear1.weight",
        "spk.blk3.tdnnd16.cam_ll.weight", "spk.transit2.conv.weight",
        "spk.transit3.bn.b", "spk.dense.weight", "spk.dense.bias",
    ]
    nbyte = 0
    for name in SAMPLE:
        t = got[name]
        arr, dtype = plan[name]
        data = bytes(t.data)
        if name == "t2w.prompt.tokens":
            want = arr.astype(np.int32).tobytes()
        elif dtype == GGML_F16 and arr.ndim >= 2:
            want = arr.astype(np.float16).tobytes()
        else:
            want = arr.astype(np.float32).tobytes()
        if data != want:
            C.fail(f"byte mismatch on {name} ({len(data)} vs {len(want)})")
        nbyte += len(data)
    print(f"byte-exact sample: {len(SAMPLE)} tensors, {nbyte} bytes OK")

    # ---- metadata -----------------------------------------------------------
    kv = kv_dict(reader)
    expect = {
        "token2wav.n_mel": 80, "token2wav.codebook": 6561,
        "token2wav.cfg_rate": 0.7, "token2wav.n_timesteps": 10,
        "token2wav.flow_d": 512, "token2wav.flow_heads": 8,
        "token2wav.flow_head_dim": 64, "token2wav.cfm_block_count": 6,
        "token2wav.cfm_up_block_count": 4, "token2wav.dit_block_count": 16,
        "token2wav.dit_hidden": 512,
    }
    for k, v in expect.items():
        got_v = kv.get(k)
        if got_v is None or abs(float(got_v) - float(v)) > 1e-6:
            C.fail(f"metadata {k}: {got_v} != {v}")
    for k in ("token2wav.prompt.tokens_len", "token2wav.prompt.mels_len",
              "token2wav.spk.fbank_len", "token2wav.rand_noise_len"):
        if k not in kv:
            C.fail(f"metadata {k} missing")
    print(f"metadata: {len(expect)} spot checks OK")

    print(f"PASS: {len(got)} tensors, plan diff empty, "
          f"file {os.path.getsize(args.gguf)} bytes")


if __name__ == "__main__":
    main()
