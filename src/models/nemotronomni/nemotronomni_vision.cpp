// nemotronomni_vision.cpp - see nemotronomni_vision.hpp for the contract.
#include "nemotronomni_vision.hpp"

#include "core/blocks/vit_block.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/media/resample.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace nemo {

static float py_round_half_even(double x) {
    return std::nearbyint(x);   // FE_TONEAREST default: ties go to even
}

void RadioTower::free_graph() {
    if (vctx) { ggml_free(vctx); vctx = nullptr; vgf = nullptr; }
    v_in = v_pos_in = v_feats = nullptr;
    v_n_tok = -1;
}

// image_processing.py _compute_target_patches (Python round() banker's
// semantics; 853x480 -> wp 54, hp 30 is the parity pin).
static void compute_target_patches(int w, int h, int min_p, int max_p, int budget,
                                   int * wp, int * hp) {
    const int P = 16;
    int ch = (int)py_round_half_even((double)h / P + 0.5);
    int cw = (int)py_round_half_even((double)w / P + 0.5);
    int patches = ch * cw;
    double factor = std::min(std::sqrt((double)budget / (double)patches), 1.0);
    int th = (int)std::floor(factor * ch);
    int tw = (int)std::floor(factor * cw);
    if (budget > min_p && th * tw < min_p) {
        const double up = std::sqrt((double)min_p / (double)(th * tw));
        th = (int)std::ceil(up * th);
        tw = (int)std::ceil(up * tw);
    }
    const int div = 2;                       // pixel-shuffle factor
    const int rem_h = th % div;
    if (rem_h) {
        const int inc = div - rem_h;
        th = ((th + inc) * tw <= budget) ? th + inc : std::max(div, th - rem_h);
    }
    const int rem_w = tw % div;
    if (rem_w) {
        const int inc = div - rem_w;
        tw = (th * (tw + inc) <= budget) ? tw + inc : std::max(div, tw - rem_w);
    }
    *wp = tw;
    *hp = th;
}

bool RadioTower::preprocess(const uint8_t * rgb, int w0, int h0,
                            std::vector<float> & pixels) {
    // budget: (max_model_len - 4) * shuffle^2, clamped into [min, max] patches
    const int budget = std::max(std::min((max_model_len - 4) * 4, max_patches), min_patches);
    compute_target_patches(w0, h0, min_patches, max_patches, budget, &grid_wp, &grid_hp);
    const int tw = grid_wp * patch, th = grid_hp * patch;

    // uint8 -> fp32 interleaved (the processor feeds a float tensor)
    std::vector<float> src((size_t)w0 * h0 * 3);
    for (size_t i = 0; i < src.size(); i++) src[i] = (float)rgb[i];

    std::vector<float> resized;
    if (w0 == tw && h0 == th) {
        resized.swap(src);
    } else {
        vqa::img::resize_bicubic_torch_aa(src.data(), h0, w0, th, tw, 3, resized);
    }

    // (x / 255 - mean) / std, CHW planes
    pixels.resize((size_t)3 * th * tw);
    for (int c = 0; c < 3; c++) {
        float * plane = pixels.data() + (size_t)c * th * tw;
        for (int y = 0; y < th; y++) {
            const float * row = resized.data() + ((size_t)y * tw) * 3;
            for (int x = 0; x < tw; x++) {
                plane[(size_t)y * tw + x] =
                    (row[(size_t)x * 3 + c] / 255.0f - img_mean[c]) / img_std[c];
            }
        }
    }
    n_tok = (grid_wp * grid_hp) / 4;
    return true;
}

// pos_embed interpolation: bilinear align_corners=False to a square
// (max(hp,wp))^2, then the window_select crop (which always lands exactly on
// (hp, wp) because the square side is the max of the two). The final
// "shape still != input_dims" interpolation of the repo never engages.
static void interp_pos_embed(RadioTower & t, int hp, int wp,
                             std::vector<float> & pos, const char * dump_dir) {
    const int side = t.n_embd > 0 ? (int)(ggml_nelements(t.pos_embd) / t.n_embd) : 0;
    const int grid = (int)std::sqrt((double)side);
    GGML_ASSERT(grid * grid == side);
    // host copy once: [tok, d] rows (ne {d, tok})
    std::vector<float> table((size_t)side * t.n_embd);
    ggml_backend_tensor_get(t.pos_embd, table.data(), 0, sizeof(float) * table.size());

    const int md = std::max(hp, wp);
    std::vector<float> sq;
    vqa::img::resize_bilinear_torch(table.data(), grid, t.n_embd, md, md, sq);
    // window_select: crop rows to hp, cols to wp
    pos.resize((size_t)hp * wp * t.n_embd);
    for (int i = 0; i < hp; i++)
        for (int j = 0; j < wp; j++)
            memcpy(pos.data() + ((size_t)i * wp + j) * t.n_embd,
                   sq.data() + ((size_t)i * md + j) * t.n_embd,
                   sizeof(float) * t.n_embd);
    if (dump_dir && *dump_dir) {
        FILE * f = nullptr;
        char pth[512];
        snprintf(pth, sizeof(pth), "%s/cpp_v_pos_enc.bin", dump_dir);
        if ((f = fopen(pth, "wb")) != nullptr) {
            fwrite(pos.data(), 4, pos.size(), f);
            fclose(f);
        }
        snprintf(pth, sizeof(pth), "%s/cpp_v_pos_enc.shape", dump_dir);
        if ((f = fopen(pth, "w")) != nullptr) { fprintf(f, "%d %d\n", hp * wp, t.n_embd); fclose(f); }
    }
}

bool RadioTower::build_graph(int n_patch, int wp, int hp) {
    free_graph();
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 32768, false);
    ggml_context * c = vctx;
    const int E = n_embd, N = n_cls + n_patch;

    v_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3 * patch * patch, n_patch);
    ggml_set_name(v_in, "v_pixels");
    ggml_set_input(v_in);
    v_pos_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, n_patch);
    ggml_set_name(v_pos_in, "v_pos_in");
    ggml_set_input(v_pos_in);

    // patch embed + learned positions, then the cls/register prefix
    ggml_tensor * x = ggml_add(c, ggml_mul_mat(c, patch_w, v_in), v_pos_in);
    ggml_set_name(x, "v_patch_out");
    x = ggml_concat(c, cls_tok, x, 1);           // [E, 10 + n_patch]

    // ---- shared ViT block: no rope, no mask, GELU(erf), LayerNorm affine
    vqa::vit::VitHookEnv venv;
    venv.gctx = c;
    venv.gf = vgf;
    venv.n_head = n_head;
    venv.n_embd = E;
    venv.N = N;
    venv.vhd = E / n_head;                       // 80
    venv.vhd_f = venv.vhd;
    venv.wo_padded = true;                       // head_dim divides exactly
    venv.no_rope = true;
    venv.dbg_names = !dump_dir.empty();
    vqa::vit::BlockHooks vh;
    vh.attention = vqa::vit::vit_attention_hook;
    vh.gelu = [](void * ud, ggml_tensor * t) -> ggml_tensor * {
        // timm's nn.GELU() (approximate="none") is the erf form
        return ggml_gelu_erf(((vqa::vit::VitHookEnv *)ud)->gctx, t);
    };
    vh.probe_layer0 = vqa::vit::vit_probe_layer0_hook;
    vh.ud = &venv;
    for (int il = 0; il < n_layer; il++) {
        vqa::vit::BlockWeights w;
        const RadioVLayer & L = vl[(size_t)il];
        w.ln1_w = L.ln1_w; w.ln1_b = L.ln1_b;
        w.qkv_w = L.wqkv; w.qkv_b = L.bqkv;
        w.wo_w = L.wo; w.wo_b = L.bo;
        w.ln2_w = L.ln2_w; w.ln2_b = L.ln2_b;
        w.fc0_w = L.fc0_w; w.fc0_b = L.fc0_b;
        w.fc1_w = L.fc1_w; w.fc1_b = L.fc1_b;
        venv.layer = il;
        x = vqa::vit::block(c, x, w, ln_eps, vh, il);
        if (!x) { fprintf(stderr, "nemotronomni vision: block %d failed\n", il); return false; }
    }
    ggml_set_name(x, "v_vit_out");               // [E, 10 + n_patch], no final norm

    // ---- drop the 10 prefix tokens, reshape to the patch grid
    ggml_tensor * fx = ggml_view_2d(c, x, E, n_patch, x->nb[1], (size_t)n_cls * x->nb[1]);
    fx = ggml_cont(c, fx);                        // [E, n_patch], token = h*wp + w
    fx = ggml_reshape_3d(c, fx, E, wp, hp);       // {E, w, h}
    // pixel_shuffle v2, scale 0.5 (both permutes; the second is what v2 adds)
    fx = ggml_reshape_3d(c, fx, 2 * E, wp / 2, hp);       // split (w, C)
    fx = ggml_cont(c, ggml_permute(c, fx, 0, 2, 1, 3));   // {2E, h, w/2}
    fx = ggml_reshape_3d(c, fx, 4 * E, hp / 2, wp / 2);   // split (h, 2C)
    fx = ggml_cont(c, ggml_permute(c, fx, 0, 2, 1, 3));   // {4E, w/2, h/2}
    fx = ggml_reshape_2d(c, fx, 4 * E, n_patch / 4);   // wp*hp/4 shuffled tokens
    ggml_set_name(fx, "v_ps_out");

    // ---- mlp1: RMSNorm(5120) -> Linear[20480, 5120] -> relu^2 -> Linear[2688, 20480]
    fx = ggml_rms_norm(c, fx, 1e-5f);
    fx = ggml_mul(c, fx, mm_ln);
    fx = ggml_mul_mat(c, mm_fc0, fx);
    fx = ggml_sqr(c, ggml_relu(c, fx));
    v_feats = ggml_mul_mat(c, mm_fc1, fx);        // [llm_dim, n_tok]
    ggml_set_name(v_feats, "v_final_embeds");
    ggml_build_forward_expand(vgf, v_feats);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_pixels", "v_pos_in", "v_patch_out",
                                           "v_blk0_attn", "v_vit_out", "v_ps_out",
                                           "v_final_embeds", nullptr };
        vqa::dump::mark_sinks(c, vgf, dn);
        vqa::dump::finalize_sinks(c, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "nemotronomni: vision graph allocation failed\n");
        return false;
    }
    v_n_tok = n_patch;     // graph keyed on the patch count (pre-shuffle)
    return true;
}

bool RadioTower::encode(ggml_backend_t backend, const std::vector<float> & pixels,
                        std::vector<float> & embeds, const char * dump) {
    dump_dir = dump ? dump : "";
    this->backend = backend;
    const int n_patch = grid_wp * grid_hp;
    if (!vgf || v_n_tok != n_patch) {
        if (!build_graph(n_patch, grid_wp, grid_hp)) return false;
    }
    if (!dump_dir.empty()) {
        std::filesystem::create_directories(dump_dir);
        FILE * f = nullptr;
        char pth[512];
        snprintf(pth, sizeof(pth), "%s/cpp_v_pixels.bin", dump_dir.c_str());
        if ((f = fopen(pth, "wb")) != nullptr) {
            fwrite(pixels.data(), 4, pixels.size(), f);
            fclose(f);
            snprintf(pth, sizeof(pth), "%s/cpp_v_pixels.shape", dump_dir.c_str());
            if ((f = fopen(pth, "w")) != nullptr) {
                const int th = grid_hp * patch, tw = grid_wp * patch;
                fprintf(f, "%d %d %d\n", 3, th, tw);
                fclose(f);
            }
        }
    }
    // patch packing: (c, yy, xx) c-major vectors, patch = row-major grid walk
    {
        std::vector<float> pin((size_t)n_patch * 3 * patch * patch);
        const int th = grid_hp * patch, tw = grid_wp * patch;
        for (int ih = 0; ih < grid_hp; ih++) {
            for (int iw = 0; iw < grid_wp; iw++) {
                float * dst = pin.data() + ((size_t)ih * grid_wp + iw) * 3 * patch * patch;
                for (int ch = 0; ch < 3; ch++)
                    for (int yy = 0; yy < patch; yy++)
                        for (int xx = 0; xx < patch; xx++)
                            dst[(size_t)ch * patch * patch + (size_t)yy * patch + xx] =
                                pixels[((size_t)ch * th + ih * patch + yy) * tw + iw * patch + xx];
            }
        }
        ggml_backend_tensor_set(v_in, pin.data(), 0, sizeof(float) * pin.size());
    }
    // positions (cached per resolution through the graph input)
    {
        std::vector<float> pos;
        interp_pos_embed(*this, grid_hp, grid_wp, pos, dump_dir.c_str());
        ggml_backend_tensor_set(v_pos_in, pos.data(), 0, sizeof(float) * pos.size());
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "nemotronomni: vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patch_out", "v_blk0_attn", "v_vit_out",
                                           "v_ps_out", "v_final_embeds", nullptr };
        for (int i = 0; dn[i]; i++) vqa::dump::write(vgf, dump_dir.c_str(), dn[i]);
    }
    embeds.resize((size_t)n_tok * llm_dim);
    ggml_backend_tensor_get(v_feats, embeds.data(), 0, sizeof(float) * embeds.size());
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "nemotronomni vision: %dx%d -> grid %dx%d, %d tokens (%.0f ms)\n",
            grid_wp * patch, grid_hp * patch, grid_wp, grid_hp, n_tok, ms);
    return true;
}

}  // namespace nemo
