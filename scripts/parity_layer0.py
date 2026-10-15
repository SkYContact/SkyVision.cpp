"""Layer-0 parity check: embedding + block 0 (dense MLP, tau attention) in torch,
using the official 3.1 math (vikhyat/moondream text.py). Dumps tensors to parity/*.npy
for comparison against moondream-cli's MD_DUMP output."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import hf_source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
ST = hf_source.find_model_safetensors()
OUT = os.path.join(ROOT, "parity")
os.makedirs(OUT, exist_ok=True)

f = safe_open(ST, framework="pt", device="cpu")
P = "text_model.transformer.h.0."

def get(name):
    return f.get_tensor(name).float()

# --- official tokenizer encode ---
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(ROOT, "tokenizer.json"))
question = "What color is an apple?"
q_ids = tk.encode(question).ids
toks = [0] + [1, 15381, 2] + q_ids + [3]
print("tokens:", toks)
T = len(toks)
np.save(os.path.join(OUT, "tokens.npy"), np.array(toks, dtype=np.int64))

# --- embeddings ---
wte = get("text_model.transformer.embd.wte.weight")      # [vocab, dim]
x = wte[toks].clone()                                    # [T, D]
np.save(os.path.join(OUT, "py_h_emb.npy"), x.numpy().astype(np.float32))

# --- block 0 ---
D, H, HD, n_kv = 2048, 32, 64, 32
ln_w = get(P + "ln.weight"); ln_b = get(P + "ln.bias")
l = F.layer_norm(x, (D,), ln_w, ln_b, 1e-5)
np.save(os.path.join(OUT, "py_l0_ln.npy"), l.numpy().astype(np.float32))

Wqkv = get(P + "mixer.Wqkv.weight"); bqkv = get(P + "mixer.Wqkv.bias")
qkv = F.linear(l, Wqkv, bqkv)                            # [T, 6144]
np.save(os.path.join(OUT, "py_l0_qkv.npy"), qkv.numpy().astype(np.float32))

# tau
tau_wq = get("text_model.transformer.h.0.tau_wq"); tau_wv = get("text_model.transformer.h.0.tau_wv")
alpha = get("text_model.transformer.h.0.tau_alpha")
tok_feat = F.gelu(qkv)
tok_q = torch.tanh(tok_feat @ tau_wq.t())                # [T, H]
pos = torch.arange(1, T + 1, dtype=torch.float32)
tau_pos = 1 + (torch.sigmoid(alpha[:, None] * pos.log()[None, :]) - 0.5)  # [H, T]

q = qkv[:, :H*HD].view(T, H, HD).transpose(0, 1)         # [H, T, HD]
k = qkv[:, H*HD:2*H*HD].view(T, n_kv, HD).transpose(0, 1)
v = qkv[:, 2*H*HD:].view(T, n_kv, HD).transpose(0, 1)
tok_v = torch.tanh(tok_feat @ tau_wv.t())
np.save(os.path.join(OUT, "py_l0_v_tau.npy"), v.numpy().astype(np.float32))
q = q * (tok_q.t() + tau_pos)[:, :, None]
v = v * (tok_v.t() + tau_pos)[:, :, None]
np.save(os.path.join(OUT, "py_l0_q_tau.npy"), q.numpy().astype(np.float32))

# rope (official 3.1: precompute_freqs_cis(dim=32); position_ids are 0-based)
theta = 1500000.0
inv = 1.0 / (theta ** (torch.arange(0, 32, 2).float() / 32))    # [16]
pos0 = torch.arange(T, dtype=torch.float32)
ang = pos0[:, None] * inv[None, :]                               # [T, 16]
fc, fs = ang.cos(), ang.sin()

def rope(t):  # t [h, T, 64]: rotate first 32 dims, non-interleaved pairs (i, i+16)
    xr = t[:, :, :32]
    a, b = xr[:, :, :16], xr[:, :, 16:32]
    out = torch.empty_like(t)
    out[:, :, :16] = a * fc[None] - b * fs[None]
    out[:, :, 16:32] = a * fs[None] + b * fc[None]
    out[:, :, 32:] = t[:, :, 32:]
    return out

q = rope(q); k = rope(k)
np.save(os.path.join(OUT, "py_l0_q_rope.npy"), q.numpy().astype(np.float32))
np.save(os.path.join(OUT, "py_l0_k_rope.npy"), k.numpy().astype(np.float32))

# attention
att = (q @ k.transpose(-1, -2)) / math.sqrt(HD)          # [H, T, T]
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
att = att.masked_fill(~mask, float("-inf"))
att = F.softmax(att, dim=-1)
o = att @ v                                              # [H, T, HD]
o = o.transpose(0, 1).reshape(T, D)
np.save(os.path.join(OUT, "py_l0_attn_out.npy"), o.numpy().astype(np.float32))

Wout = get(P + "mixer.out_proj.weight"); bout = get(P + "mixer.out_proj.bias")
attn_out = F.linear(o, Wout, bout)
np.save(os.path.join(OUT, "py_l0_attn_proj.npy"), attn_out.numpy().astype(np.float32))

# dense MLP: gelu tanh approx
fc1w = get(P + "mlp.fc1.weight"); fc1b = get(P + "mlp.fc1.bias")
fc2w = get(P + "mlp.fc2.weight"); fc2b = get(P + "mlp.fc2.bias")
mlp_out = F.linear(F.gelu(F.linear(l, fc1w, fc1b), approximate="tanh"), fc2w, fc2b)
np.save(os.path.join(OUT, "py_l0_mlp.npy"), mlp_out.numpy().astype(np.float32))

x1 = x + attn_out + mlp_out
np.save(os.path.join(OUT, "py_h_L0.npy"), x1.numpy().astype(np.float32))
print("wrote parity tensors for T =", T)
