"""Convert the MiniCPM-o 4.5 CosyVoice2 token2wav vocoder assets to GGUF.

Usage::

  python convert/convert_minicpmo45_vocoder.py [--src DIR] [--dump DIR]
                                               [-o OUT.gguf]
                                               [--out-type f16|f32]

Sources (E:/SkyVision-weights/minicpmo45/assets/token2wav):
  flow.pt  (623 MB fp32) CosyVoice2 CausalMaskedDiffWithXvec: token embedding
           + UpsampleConformerEncoderV2 (6+4 rel-pos conformer blocks,
           nearest-x2 upsample conv k5) + DiT-16 CFM estimator (adaLN-zero,
           qk-norm, causal-conv branch).
  hift.pt  (83 MB fp32) HiFTGenerator: ConvRNNF0Predictor + ConvTranspose1d
           stack (8/5/3) + Snake resblocks + NSF sine source + ISTFT head.
  campplus.onnx (28 MB fp32) CAM++ speaker encoder: conv2d head (3 stride-2
           stages, freq 80->10), tdnn 320->128 k5 s2, 3 xvector blocks of
           12/24/16 tdnnd+CAM units, transit 1x1 convs, attentive stats
           (mean/unbiased-std), dense 1024->192.

Prompt-side artifacts come from the official-pipeline dump at
D:/S2TT/minicpmo45-port/4vocoder (dump_vocoder.py / dump_randnoise.py):
speech-tokenizer codes of the fixed reference voice (HT_ref_audio.wav), its
kaldi-fbank CAM++ input, the replicate-padded prompt mel, and the CFM
rand_noise module buffer. The 496 MB speech_tokenizer onnx is deliberately
NOT ported - the engine vocoder consumes its frozen output for the fixed
prompt voice.

Output: minicpmo-vocoder.gguf, general.architecture = "token2wav".
Namespaces (no collision with the main GGUF's blk./v./a./rs./tk.):
  t2w.   prompt cache + rand_noise buffer
  cfm.   flow encoder (conformer) + projections
  dit.   DiT CFM estimator
  hifi.  HiFT generator
  spk.   campplus (all xvector BNs standalone as full bn.{w,b,m,v} sets;
         only the dense BN is folded into the dense conv; head convs arrive
         pre-folded from the ONNX export)

Precision contract (mirrors convert_minicpmo45.py): matmul/conv operands F16
(--out-type f32 keeps tensors F32); every norm/bias/affine/pos table F32.
Conv layouts: Conv1d/Conv2d keep the torch memory layout (== the ggml kernel
order [K, IC, OC] / [KW, KH, IC, OC]); ConvTranspose1d weights are
transposed [IC,OC,K] -> [K,OC,IC] (ggml conv_transpose_1d order).
Weight-norm parametrizations are recombined: w = g * v / ||v||_channel.
Linear weights are transposed to [in, out] (ggml mul_mat orientation).
"""
import argparse
import os
import sys

import numpy as np
import torch
import gguf

SRC = "E:/SkyVision-weights/minicpmo45/assets/token2wav"
DUMP = "D:/S2TT/minicpmo45-port/4vocoder"
DEFAULT_OUT = "E:/SkyVision-weights/minicpmo45/minicpmo-vocoder.gguf"

OUTTYPES = {"f16": gguf.LlamaFileType.MOSTLY_F16,
            "f32": gguf.LlamaFileType.ALL_F32}
BN_EPS = 1e-5


def log(msg):
    print(msg, flush=True)


def fail(msg, code=1):
    print("FAIL: " + msg, flush=True)
    sys.exit(code)


def to_np(t):
    return t.detach().numpy().astype(np.float32, copy=True)


# ---------------------------------------------------------------- sources
class Sources:
    def __init__(self, src):
        self.flow = torch.load(f"{src}/flow.pt", map_location="cpu",
                               weights_only=True)
        log(f"flow.pt: {len(self.flow)} tensors")
        hift = torch.load(f"{src}/hift.pt", map_location="cpu",
                          weights_only=True)
        self.hift = {k.replace("generator.", ""): v for k, v in hift.items()}
        log(f"hift.pt: {len(self.hift)} tensors")
        self.used = set()

        import onnx
        from onnx import numpy_helper
        self.onnx_model = onnx.load(f"{src}/campplus.onnx")
        self.cp = {}
        for t in self.onnx_model.graph.initializer:
            self.cp[t.name] = numpy_helper.to_array(t)
        for n in self.onnx_model.graph.node:
            if n.op_type != "Constant":
                continue
            for a in n.attribute:
                if a.name == "value" and \
                        getattr(a.t, "elem_type", 0) == onnx.TensorProto.FLOAT:
                    self.cp[n.output[0]] = np.frombuffer(
                        a.t.raw_data, dtype=np.float32).reshape(a.t.dims).copy()
        self.convs = {}                       # node name -> (w, b) initializer
        for n in self.onnx_model.graph.node:
            if n.op_type == "Conv":
                self.convs[n.name] = (n.input[1],
                                      n.input[2] if len(n.input) > 2 else None)
        log(f"campplus.onnx: {len(self.cp)} initializers, "
            f"{len(self.convs)} conv nodes")

    def f(self, sd, name, shape=None):
        self.used.add(name)
        t = sd[name]
        if shape is not None and tuple(t.shape) != tuple(shape):
            fail(f"{name}: shape {tuple(t.shape)} != planned {tuple(shape)}")
        return to_np(t)

    def cp_t(self, name, shape=None):
        self.used.add(name)
        arr = self.cp[name]
        if shape is not None and tuple(arr.shape) != tuple(shape):
            fail(f"campplus {name}: shape {arr.shape} != planned {shape}")
        return arr.astype(np.float32, copy=True)

    def cp_conv(self, node, bn=None):
        """weights of an onnx conv node, torch layout, BN folded into bias."""
        wname, bname = self.convs[node]
        w = self.cp_t(wname)
        b = self.cp_t(bname) if bname else None
        if bn is not None:
            g, beta, mean, var = bn
            s = g / np.sqrt(var + BN_EPS)
            if b is None:
                b = np.zeros_like(mean)
            b = (b - mean) * s + beta
            w = w * s.reshape(-1, *([1] * (w.ndim - 1)))
        return w, b

    def wn(self, prefix):
        """weight_norm recombine -> [OC, IC, K] (torch conv layout)."""
        g = self.f(self.hift, prefix + ".parametrizations.weight.original0")
        v = self.f(self.hift, prefix + ".parametrizations.weight.original1")
        w = g.reshape(-1, 1, 1) * v / np.sqrt(
            (v * v).sum(axis=(1, 2), keepdims=True))
        return w.astype(np.float32)

    def bn(self, pfx):
        return (self.cp_t(f"{pfx}.weight"), self.cp_t(f"{pfx}.bias"),
                self.cp_t(f"{pfx}.running_mean"),
                self.cp_t(f"{pfx}.running_var"))


# ---------------------------------------------------------------- writer
class W:
    def __init__(self, gw, f16):
        self.gw = gw
        self.f16 = f16

    def t(self, name, arr, force32=False):
        arr = np.ascontiguousarray(arr)
        if self.f16 and arr.ndim >= 2 and not force32:
            self.gw.add_tensor(name, arr.astype(np.float16))
        else:
            self.gw.add_tensor(name, arr.astype(np.float32))

    def lin(self, name, w_oi, b=None):
        """torch nn.Linear weight [out, in] stored AS-IS - the gguf ne
        reversal turns it into the ggml [in, out] mul_mat orientation."""
        self.t(name + ".weight", w_oi)
        if b is not None:
            self.gw.add_tensor(name + ".bias", b.astype(np.float32))

    def conv(self, name, w, b=None):
        # 1-D conv kernels: torch [OC, IC, K] flattened per output channel to
        # [OC, IC*K] -> gguf ne = [IC*K, OC]; the engine contracts over ne[0]
        # (rows are (ic, k) ic-major, matching ggml_im2col columns).
        if w.ndim == 3:
            w = w.reshape(w.shape[0], -1)   # [OC, IC*K] (1-D convs only)
        self.t(name + ".weight", w)
        if b is not None:
            self.gw.add_tensor(name + ".bias", b.astype(np.float32))

    def conv3d(self, name, w, b=None):
        # keep the torch [OC, IC, K] layout for ggml_im2col (ne [K, IC, OC])
        self.t(name + ".weight", w)
        if b is not None:
            self.gw.add_tensor(name + ".bias", b.astype(np.float32))


# ---------------------------------------------------------------- convert
def convert(args):
    S = Sources(args.src)
    flow, hift = S.flow, S.hift
    gw = gguf.GGUFWriter(args.out, "token2wav")
    gw.add_architecture()
    gw.add_name("MiniCPM-o-4.5-vocoder")
    gw.add_file_type(OUTTYPES[args.out_type])
    gw.add_uint32("general.quantization_version", 2)
    w = W(gw, args.out_type == "f16")

    # ---------------- prompt cache + rand_noise -----------------------------
    # The CFM noise lives in the rand_noise module buffer created at Token2wav
    # init time; it is NOT reproducible across runs (RNG state at init differs
    # with import/init order). The authoritative slice is the one captured
    # mid-pipeline (z, same run as every gate artifact). The GGUF buffer is
    # z-prefixed; columns beyond the oracle length continue from a standalone
    # torch.randn draw (production-only region, no parity claim).
    z = np.load(f"{args.dump}/z.npy")                     # [1, 80, 506]
    z = z[0]                                              # [80, 506]
    tail = np.load(f"{args.dump}/rand_noise_full.npy")    # [80, 30000]
    n = z.shape[1]
    if tail.shape[1] < n:
        fail("rand_noise_full.npy shorter than the dumped z slice")
    buf = np.concatenate([z, tail[:, n:]], axis=1)
    gw.add_tensor("t2w.rand_noise", buf.astype(np.float32))
    pt = np.load(f"{args.dump}/prompt_speech_tokens.npy").reshape(-1)
    gw.add_tensor("t2w.prompt.tokens", pt.astype(np.int32))
    pm = np.load(f"{args.dump}/prompt_mels.npy")          # [1, T2p, 80]
    gw.add_tensor("t2w.prompt.mels",
                  pm.reshape(pm.shape[1], 80).astype(np.float32))
    fb = np.load(f"{args.dump}/spk_fbank.npy")            # [1, T, 80]
    gw.add_tensor("t2w.spk.fbank",
                  fb.reshape(fb.shape[1], 80).astype(np.float32))
    emb_ref = np.load(f"{args.dump}/spk_emb_ort.npy").reshape(-1)
    gw.add_tensor("t2w.spk.emb_ref", emb_ref.astype(np.float32))

    gw.add_uint32("token2wav.prompt.tokens_len", int(pt.size))
    gw.add_uint32("token2wav.prompt.mels_len", int(pm.shape[1]))
    gw.add_uint32("token2wav.spk.fbank_len", int(fb.shape[1]))
    gw.add_uint32("token2wav.rand_noise_len", int(buf.shape[1]))
    gw.add_uint32("token2wav.n_mel", 80)
    gw.add_uint32("token2wav.codebook", 6561)
    gw.add_float32("token2wav.cfg_rate", 0.7)
    gw.add_uint32("token2wav.n_timesteps", 10)
    gw.add_uint32("token2wav.flow_d", 512)
    gw.add_uint32("token2wav.flow_heads", 8)
    gw.add_uint32("token2wav.flow_head_dim", 64)
    gw.add_uint32("token2wav.cfm_block_count", 6)
    gw.add_uint32("token2wav.cfm_up_block_count", 4)
    gw.add_uint32("token2wav.dit_block_count", 16)
    gw.add_uint32("token2wav.dit_hidden", 512)
    gw.add_float32("token2wav.dit_eps", 1e-6)
    gw.add_float32("token2wav.dit_qk_eps", 1e-5)
    gw.add_float32("token2wav.conformer_ln_eps", 1e-12)
    gw.add_float32("token2wav.embed_ln_eps", 1e-5)

    # ================= flow (CausalMaskedDiffWithXvec) ======================
    w.t("cfm.input_emb.weight", S.f(flow, "input_embedding.weight"))
    w.lin("cfm.spk_proj", S.f(flow, "spk_embed_affine_layer.weight"),
          S.f(flow, "spk_embed_affine_layer.bias"))
    w.lin("cfm.embed", S.f(flow, "encoder.embed.out.0.weight"),
          S.f(flow, "encoder.embed.out.0.bias"))
    gw.add_tensor("cfm.embed_ln.weight",
                  S.f(flow, "encoder.embed.out.1.weight"))
    gw.add_tensor("cfm.embed_ln.bias",
                  S.f(flow, "encoder.embed.out.1.bias"))
    w.conv3d("cfm.lookahead.conv1",
             S.f(flow, "encoder.pre_lookahead_layer.conv1.weight"),
             S.f(flow, "encoder.pre_lookahead_layer.conv1.bias"))
    w.conv3d("cfm.lookahead.conv2",
             S.f(flow, "encoder.pre_lookahead_layer.conv2.weight"),
             S.f(flow, "encoder.pre_lookahead_layer.conv2.bias"))

    def conformer_block(gname, mod, i):
        b = f"cfm.{gname}{i}"
        for a, h in (("ln_mha", "norm_mha"), ("ln_ff", "norm_ff")):
            gw.add_tensor(f"{b}.{a}.weight",
                          S.f(flow, f"encoder.{mod}.{i}.{h}.weight"))
            gw.add_tensor(f"{b}.{a}.bias",
                          S.f(flow, f"encoder.{mod}.{i}.{h}.bias"))
        base = f"encoder.{mod}.{i}.self_attn"
        for nm, h in (("wq", "linear_q"), ("wk", "linear_k"),
                      ("wv", "linear_v"), ("wo", "linear_out"),
                      ("wpos", "linear_pos")):
            bias = None if h == "linear_pos" else \
                S.f(flow, f"{base}.{h}.bias")
            w.lin(f"{b}.{nm}", S.f(flow, f"{base}.{h}.weight"), bias)
        gw.add_tensor(f"{b}.pos_bias_u",
                      S.f(flow, f"{base}.pos_bias_u").reshape(-1))
        gw.add_tensor(f"{b}.pos_bias_v",
                      S.f(flow, f"{base}.pos_bias_v").reshape(-1))
        w.lin(f"{b}.ffn_gate",
              S.f(flow, f"encoder.{mod}.{i}.feed_forward.w_1.weight"),
              S.f(flow, f"encoder.{mod}.{i}.feed_forward.w_1.bias"))
        w.lin(f"{b}.ffn_down",
              S.f(flow, f"encoder.{mod}.{i}.feed_forward.w_2.weight"),
              S.f(flow, f"encoder.{mod}.{i}.feed_forward.w_2.bias"))

    for i in range(6):
        conformer_block("blk.", "encoders", i)
    w.conv3d("cfm.up.conv",
             S.f(flow, "encoder.up_layer.conv.weight"),
             S.f(flow, "encoder.up_layer.conv.bias"))
    w.lin("cfm.up_embed", S.f(flow, "encoder.up_embed.out.0.weight"),
          S.f(flow, "encoder.up_embed.out.0.bias"))
    gw.add_tensor("cfm.up_embed_ln.weight",
                  S.f(flow, "encoder.up_embed.out.1.weight"))
    gw.add_tensor("cfm.up_embed_ln.bias",
                  S.f(flow, "encoder.up_embed.out.1.bias"))
    for i in range(4):
        conformer_block("ublkc.", "up_encoders", i)
    gw.add_tensor("cfm.after_norm.weight", S.f(flow, "encoder.after_norm.weight"))
    gw.add_tensor("cfm.after_norm.bias", S.f(flow, "encoder.after_norm.bias"))
    w.lin("cfm.proj", S.f(flow, "encoder_proj.weight"),
          S.f(flow, "encoder_proj.bias"))

    # ---------------- DiT ---------------------------------------------------
    est = "decoder.estimator"
    w.lin("dit.t_mlp0", S.f(flow, f"{est}.t_embedder.mlp.0.weight"),
          S.f(flow, f"{est}.t_embedder.mlp.0.bias"))
    w.lin("dit.t_mlp1", S.f(flow, f"{est}.t_embedder.mlp.2.weight"),
          S.f(flow, f"{est}.t_embedder.mlp.2.bias"))
    w.lin("dit.in_proj", S.f(flow, f"{est}.in_proj.weight"),
          S.f(flow, f"{est}.in_proj.bias"))
    for i in range(16):
        b = f"dit.blk.{i}"
        p = f"{est}.blocks.{i}"
        for nm, h in (("wq", "to_q"), ("wk", "to_k"), ("wv", "to_v"),
                      ("wo", "proj")):
            w.lin(f"{b}.{nm}", S.f(flow, f"{p}.attn.{h}.weight"),
                  S.f(flow, f"{p}.attn.{h}.bias"))
        for nm in ("q_norm", "k_norm"):
            gw.add_tensor(f"{b}.{nm}.weight",
                          S.f(flow, f"{p}.attn.{nm}.weight"))
            gw.add_tensor(f"{b}.{nm}.bias",
                          S.f(flow, f"{p}.attn.{nm}.bias"))
        w.lin(f"{b}.mlp_gate", S.f(flow, f"{p}.mlp.fc1.weight"),
              S.f(flow, f"{p}.mlp.fc1.bias"))
        w.lin(f"{b}.mlp_down", S.f(flow, f"{p}.mlp.fc2.weight"),
              S.f(flow, f"{p}.mlp.fc2.bias"))
        w.conv(f"{b}.conv1", S.f(flow, f"{p}.conv.block.1.weight"),
               S.f(flow, f"{p}.conv.block.1.bias"))
        gw.add_tensor(f"{b}.conv_ln.weight",
                      S.f(flow, f"{p}.conv.block.3.weight"))
        gw.add_tensor(f"{b}.conv_ln.bias",
                      S.f(flow, f"{p}.conv.block.3.bias"))
        w.conv(f"{b}.conv2", S.f(flow, f"{p}.conv.block.6.weight"),
               S.f(flow, f"{p}.conv.block.6.bias"))
        w.lin(f"{b}.adaln", S.f(flow, f"{p}.adaLN_modulation.1.weight"),
              S.f(flow, f"{p}.adaLN_modulation.1.bias"))
    w.lin("dit.fin.adaln",
          S.f(flow, f"{est}.final_layer.adaLN_modulation.1.weight"),
          S.f(flow, f"{est}.final_layer.adaLN_modulation.1.bias"))
    w.lin("dit.fin.out", S.f(flow, f"{est}.final_layer.linear.weight"),
          S.f(flow, f"{est}.final_layer.linear.bias"))

    # ================= hift (HiFTGenerator) =================================
    for i, ci in enumerate((0, 2, 4, 6, 8)):
        w.conv(f"hifi.f0.c{i}", S.wn(f"f0_predictor.condnet.{ci}"),
               S.f(hift, f"f0_predictor.condnet.{ci}.bias"))
    w.lin("hifi.f0.head", S.f(hift, "f0_predictor.classifier.weight"),
          S.f(hift, "f0_predictor.classifier.bias"))
    w.lin("hifi.src_linear", S.f(hift, "m_source.l_linear.weight"),
          S.f(hift, "m_source.l_linear.bias"))
    w.conv3d("hifi.conv_pre", S.wn("conv_pre"), S.f(hift, "conv_pre.bias"))
    for i in range(3):
        # torch ConvTranspose1d [IC, OC, K] memory == the ggml conv_transpose_1d
        # kernel order ne=[K, OC, IC] - keep as-is (the wn() norm is over
        # (OC, K) per IC because original0 is [IC, 1, 1] for these layers)
        w.conv(f"hifi.up.{i}", S.wn(f"ups.{i}"),
               S.f(hift, f"ups.{i}.bias"))
    ri = 0
    for stage in range(3):
        for j in range(3):
            for c in (1, 2):
                for k in range(3):
                    w.conv(f"hifi.rb.{ri}.c{c}.{k}",
                           S.wn(f"resblocks.{ri}.convs{c}.{k}"),
                           S.f(hift, f"resblocks.{ri}.convs{c}.{k}.bias"))
            for k in range(3):
                for half in (1, 2):
                    gw.add_tensor(
                        f"hifi.rb.{ri}.a{half}.{k}",
                        S.f(hift, f"resblocks.{ri}.activations{half}.{k}.alpha"))
            ri += 1
    for i in range(3):
        w.conv(f"hifi.sdown.{i}", S.f(hift, f"source_downs.{i}.weight"),
               S.f(hift, f"source_downs.{i}.bias"))
    for j in range(3):
        for c in (1, 2):
            for k in range(3):
                w.conv(f"hifi.srb.{j}.c{c}.{k}",
                       S.wn(f"source_resblocks.{j}.convs{c}.{k}"),
                       S.f(hift, f"source_resblocks.{j}.convs{c}.{k}.bias"))
        for k in range(3):
            for half in (1, 2):
                gw.add_tensor(
                    f"hifi.srb.{j}.a{half}.{k}",
                    S.f(hift, f"source_resblocks.{j}.activations{half}.{k}.alpha"))
    w.conv3d("hifi.conv_post", S.wn("conv_post"), S.f(hift, "conv_post.bias"))

    # ================= campplus (CAM++) =====================================
    M = S.onnx_model.graph
    # BN plan: every xvector BN (unit entries + transits) sits on a relu or
    # concat output -> NOTHING folds into the convs; each is stored as a full
    # parameter set bn.{w,b,m,v} (gamma, beta, running_mean, running_var).
    # Only the dense BN (immediately after the dense conv, before the output)
    # is folded into the dense conv. The head convs arrive pre-folded from
    # the ONNX export itself.

    # ---- head (conv2d, torch [OC, IC, KH, KW] kept) ----
    hd = "spk.head"
    w.conv(f"{hd}.conv1", *S.cp_conv("/head/conv1/Conv"))
    w.conv("spk.layer1.0.conv1", *S.cp_conv("/head/layer1/layer1.0/conv1/Conv"))
    w.conv("spk.layer1.0.conv2", *S.cp_conv("/head/layer1/layer1.0/conv2/Conv"))
    w.conv("spk.layer1.0.shortcut", *S.cp_conv("/head/layer1/layer1.0/shortcut/shortcut.0/Conv"))
    w.conv("spk.layer1.1.conv1", *S.cp_conv("/head/layer1/layer1.1/conv1/Conv"))
    w.conv("spk.layer1.1.conv2", *S.cp_conv("/head/layer1/layer1.1/conv2/Conv"))
    w.conv("spk.layer2.0.conv1", *S.cp_conv("/head/layer2/layer2.0/conv1/Conv"))
    w.conv("spk.layer2.0.conv2", *S.cp_conv("/head/layer2/layer2.0/conv2/Conv"))
    w.conv("spk.layer2.0.shortcut", *S.cp_conv("/head/layer2/layer2.0/shortcut/shortcut.0/Conv"))
    w.conv("spk.layer2.1.conv1", *S.cp_conv("/head/layer2/layer2.1/conv1/Conv"))
    w.conv("spk.layer2.1.conv2", *S.cp_conv("/head/layer2/layer2.1/conv2/Conv"))
    w.conv("spk.conv2", *S.cp_conv("/head/conv2/Conv"))
    # tdnn 320->128 k5 s2 p2 (raw conv; entry BN is post-relu, kept separate)
    w.conv3d("spk.tdnn", *S.cp_conv("/xvector/tdnn/linear/Conv"))

    def cp_bn(prefix, name):
        g, beta, mean, var = S.bn(name)
        w.t(prefix + ".bn.w", g)
        w.t(prefix + ".bn.b", beta)
        w.t(prefix + ".bn.m", mean)
        w.t(prefix + ".bn.v", var)

    cp_bn("spk.blk1.tdnnd1", "xvector.block1.tdnnd1.nonlinear1.batchnorm")
    cp_bn("spk.blk2.tdnnd1", "xvector.block2.tdnnd1.nonlinear1.batchnorm")
    cp_bn("spk.blk3.tdnnd1", "xvector.block3.tdnnd1.nonlinear1.batchnorm")

    # ---- xvector blocks ----
    NB = {1: 12, 2: 24, 3: 16}
    for bi in (1, 2, 3):
        for ti in range(1, NB[bi] + 1):
            p = f"/xvector/block{bi}/tdnnd{ti}"
            b = f"spk.blk{bi}.tdnnd{ti}"
            if ti >= 2:   # entry BN normalizes the running concat tensor
                cp_bn(b, f"xvector.block{bi}.tdnnd{ti}.nonlinear1.batchnorm")
            wname, bname = S.convs[f"{p}/linear1/Conv"]
            w.lin(f"{b}.linear1", S.cp_t(wname).squeeze(-1),
                  S.cp_t(bname))                                   # [128, Ci]
            wname, _ = S.convs[f"{p}/cam_layer/linear_local/Conv"]
            w.conv3d(f"{b}.cam_ll", S.cp_t(wname))   # [32,128,3], no bias

            wname, bname = S.convs[f"{p}/cam_layer/linear1/Conv"]
            w.lin(f"{b}.cam1", S.cp_t(wname).squeeze(-1), S.cp_t(bname))
            wname, bname = S.convs[f"{p}/cam_layer/linear2/Conv"]
            w.lin(f"{b}.cam2", S.cp_t(wname).squeeze(-1), S.cp_t(bname))

    # ---- transits: BN(cat) -> relu -> conv 1x1 (raw; the follower BN of the
    # next block is also post-conv and stored standalone) ----
    transit_nodes = {n.name.split("/")[2]: n for n in M.node
                     if n.op_type == "Conv" and "/transit" in n.name
                     and n.name.endswith("linear/Conv")}
    for ti in (1, 2, 3):
        n = transit_nodes[f"transit{ti}"]
        wt, bt = S.cp_conv(n.name)               # raw conv, no fold
        w.lin(f"spk.transit{ti}.conv", wt.reshape(wt.shape[0], -1), bt)  # 1x1
        cp_bn(f"spk.transit{ti}", f"xvector.transit{ti}.nonlinear.batchnorm")

    # ---- dense: stats(1024) -> conv1x1 192 (dense BN folded) ----
    wname, bname = S.convs["/xvector/dense/linear/Conv"]
    # the dense BN's gamma/beta are Constant nodes (ones/zeros), only
    # running_mean/var are initializers
    mean = S.cp_t("xvector.dense.nonlinear.batchnorm.running_mean")
    var = S.cp_t("xvector.dense.nonlinear.batchnorm.running_var")
    g, beta = np.ones_like(mean), np.zeros_like(mean)
    s = g / np.sqrt(var + BN_EPS)
    wd = S.cp_t(wname).reshape(192, 1024) * s.reshape(-1, 1)
    w.t("spk.dense.weight", wd)                      # torch [out, in] as-is
    gw.add_tensor("spk.dense.bias", (beta - mean * s).astype(np.float32))

    # ---------------- coverage ---------------------------------------------
    for nm, sd in (("flow", flow), ("hift", hift), ("campplus", S.cp)):
        unused = [k for k in sd if k not in S.used]
        if unused:
            fail(f"{nm} unconsumed initializers ({len(unused)}): {unused[:8]}")
    log(f"coverage: flow {len(flow)}, hift {len(hift)}, "
        f"campplus {len(S.cp)} - all consumed, diff set empty")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    log(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    ap.add_argument("--out-type", default="f16", choices=list(OUTTYPES))
    convert(ap.parse_args())
