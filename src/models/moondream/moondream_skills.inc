// moondream_skills.inc - the moondream3.1 skills shared by every front end.
// Included from src/models/moondream/moondream.cpp after Engine/VisionModel/RegionModel.
//
//   md_prefill_image / md_ensure_img_prefill  image -> [bos; 729 img tokens] KV prefix
//   md_cache                                  that prefix, so repeated questions skip the encode
//   md_detect / md_point / md_gaze            official _generate_points / _detect_gaze
//   md_tok_embs                               token -> model-dim embeddings (gaze prompt)
//
// The embedded HTTP server that used to live here is gone: `vqa-cli --serve`
// (src/cli/vqa_server.cpp) drives this backend through vqa::Model like every other
// model, and it already knows how to discover moondream + its *-vision.gguf.o external dependencies — hand-rolled HTTP on
// plain sockets, sequential request handling (single local user).
//
//   GET  /                    → single-page UI (embedded HTML)
//   POST /api/ask             → body {"prompt": "...", "n": 128} → {"text": "..."}
//   POST /api/vision?q=...&mode=ask|detect&obj=...
//        body = raw image bytes (PNG/JPEG, decoded by stb_image) → {"text": "..."}
//        or {"boxes": [[x0,y0,x1,y1], ...]} (normalized, for mode=detect)


#include <map>
#include <string_view>
#include <array>
#include <sstream>



// tiny JSON "string field" extractor: finds "key":"value" (no escapes in value assumed)


// ---- generation helpers (shared by /api/ask and /api/vision) ----


// image → [bos;729] embedding prefix → prefill (bidirectional) → prompt prefill
static void md_prefill_image(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h,
                             std::vector<float> & prefix_emb) {
    int th = 0, tw = 0, n_crops = 0;
    std::vector<float> crops = mdprep::prepare_crops(rgb, w, h, &th, &tw, &n_crops);
    std::vector<float> enc = vm.encode_crops(crops.data(), n_crops, vm.n_layers, e.backend);
    std::vector<float> img_emb = vm.project(enc.data(), n_crops, th, tw, e.backend);
    const int D = e.model.n_embd;
    std::vector<ggml_fp16_t> bos16(D);
    ggml_backend_tensor_get(e.model.tok_embd, bos16.data(), 0, sizeof(ggml_fp16_t) * D);
    prefix_emb.resize((size_t)(1 + 729) * D);
    for (int d = 0; d < D; d++) prefix_emb[d] = ggml_fp16_to_fp32(bos16[d]);
    memcpy(prefix_emb.data() + D, img_emb.data(), sizeof(float) * (size_t)729 * D);
    e.step({}, nullptr, prefix_emb.data(), 730, nullptr, true);
}

// ---- cross-request caches ---------------------------------------------------
// same-image cache: repeating queries on one image (the typical web UI flow)
// skip both the SigLIP encode AND the 730-token prefill by rewinding n_past
// to 730 — the KV slots beyond it are simply overwritten by the next prompt.
// same-question cache: repeated BPE encode of identical question text.
struct MdReqCache {
    uint64_t img_key = 0;             // FNV-1a of raw image body
    std::vector<float> prefix;        // [730, D] bos+img embeddings
    bool kv_valid = false;            // engine KV currently holds this prefix
    uint64_t q_key = 0;
    std::vector<int> q_toks;
};
static MdReqCache md_cache;

static uint64_t md_fnv(std::string_view s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) { h ^= (unsigned char)c; h *= 1099511628211ull; }
    return h;
}

// cache key for an image body (geometry folded in); shared by the prefill paths
static uint64_t md_img_key(const uint8_t * rgb, int w, int h) {
    uint64_t key = md_fnv(std::string_view((const char *)rgb, (size_t)w * h * 3));
    key ^= (uint64_t)w << 32 ^ (uint64_t)h;
    return key;
}

// make sure the engine holds THIS image's prefix at KV [0,730); n_past ends at 730
static void md_ensure_img_prefill(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h) {
    // the key must actually gate the reuse: the web UI sends a different image
    // per request, and reusing the first image's prefix made every answer
    // describe the old picture
    uint64_t key = md_img_key(rgb, w, h);
    if (md_cache.img_key == key && md_cache.kv_valid) { e.n_past = 730; return; }   // rewind: reuse KV prefix
    if (md_cache.img_key == key && !md_cache.prefix.empty()) {
        e.step({}, nullptr, md_cache.prefix.data(), 730, nullptr, true);   // engine was reset, prefix hot
        md_cache.kv_valid = true;
        return;
    }
    e.reset();   // different image: clear the old prefix before re-encoding
    md_cache.prefix.clear();
    md_prefill_image(e, vm, rgb, w, h, md_cache.prefix);   // encode + prefill + fill cache
    md_cache.img_key = key;
    md_cache.q_key = 0;   // question cache is only valid within one image
    md_cache.kv_valid = true;
}


static std::vector<std::array<float,4>> md_detect(Engine & e, VisionModel & vm, RegionModel & rm,
                                                  const uint8_t * rgb, int w, int h,
                                                  const std::string & obj) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    std::vector<int> dtoks = { 1, 7235, 476, 2 };
    std::string o = " " + obj;
    auto oq = e.tok.encode(o);
    dtoks.insert(dtoks.end(), oq.begin(), oq.end());
    dtoks.push_back(3);
    const int D = e.model.n_embd;
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int next = e.step(dtoks, lv.data(), nullptr, 0, hidden.data(), false);
    next = (lv[0] >= lv[5]) ? 0 : 5;
    std::vector<std::array<float,4>> boxes;
    while (next != 0 && boxes.size() < 20) {
        float xl[1024];
        rm.decode_coordinate(hidden.data(), xl);
        float xc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&xc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        rm.decode_coordinate(hidden.data(), xl);
        float yc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&yc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        float sl[2048];
        rm.decode_size(hidden.data(), sl);
        int bins[2] = { vqa::greedy_argmax(sl, 1024), vqa::greedy_argmax(sl + 1024, 1024) };
        float sizes[2];
        RegionModel::bins_to_size(bins, sizes);
        rm.encode_size(sizes, 1, emb.data());
        // same single-step fix as md_point: logits and hidden come from one call
        e.step({}, lv.data(), emb.data(), 1, hidden.data(), false);
        boxes.push_back({{ xc - sizes[0] / 2, yc - sizes[1] / 2, xc + sizes[0] / 2, yc + sizes[1] / 2 }});
        next = (lv[0] >= lv[5]) ? 0 : 5;
    }
    return boxes;
}

// rows of the token embedding table -> raw f32 embeddings (f16 table)
static void md_tok_embs(Engine & e, const std::vector<int> & toks, std::vector<float> & out) {
    const int D = e.model.n_embd;
    out.resize(toks.size() * D);
    std::vector<ggml_fp16_t> row(D);
    for (size_t i = 0; i < toks.size(); i++) {
        ggml_backend_tensor_get(e.model.tok_embd, row.data(),
                                (size_t)toks[i] * D * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t) * D);
        for (int d = 0; d < D; d++) out[i * D + d] = ggml_fp16_to_fp32(row[d]);
    }
}

// official point skill: template [1,2581,2] + " <obj>" + [3], no size, N points
static std::vector<std::array<float,2>> md_point(Engine & e, VisionModel & vm, RegionModel & rm,
                                                 const uint8_t * rgb, int w, int h,
                                                 const std::string & obj) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    std::vector<int> dtoks = { 1, 2581, 2 };
    std::string o = " " + obj;
    auto oq = e.tok.encode(o);
    dtoks.insert(dtoks.end(), oq.begin(), oq.end());
    dtoks.push_back(3);
    const int D = e.model.n_embd;
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int next = e.step(dtoks, lv.data(), nullptr, 0, hidden.data(), false);
    next = (lv[0] >= lv[5]) ? 0 : 5;
    std::vector<std::array<float,2>> pts;
    while (next != 0 && pts.size() < 20) {
        float xl[1024];
        rm.decode_coordinate(hidden.data(), xl);
        float xc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&xc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        rm.decode_coordinate(hidden.data(), xl);
        float yc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&yc, 1, emb.data());
        // ONE step yields both the next hidden and the eos/coord logits - the
        // old code fed the y-embedding twice (once for hidden, once for logits),
        // shifting every following position and producing marching garbage points
        e.step({}, lv.data(), emb.data(), 1, hidden.data(), false);
        pts.push_back({{ xc, yc }});
        next = (lv[0] >= lv[5]) ? 0 : 5;
    }
    return pts;
}

// official detect_gaze skill: embedding-level prompt
//   emb("\n\nPoint:") + enc(x) + enc(y) + emb(" gaze\n\n")
// then eos -> none, else one predicted (x,y) fixation point
static bool md_gaze(Engine & e, VisionModel & vm, RegionModel & rm,
                    const uint8_t * rgb, int w, int h, float sx, float sy,
                    float & gx, float & gy) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    const int D = e.model.n_embd;
    std::vector<float> a, b;
    md_tok_embs(e, e.tok.encode("\n\nPoint:"), a);
    md_tok_embs(e, e.tok.encode(" gaze\n\n"), b);
    std::vector<float> xemb(D), yemb(D);   // encode_coordinate projects into model dim (2048)
    rm.encode_coordinate(&sx, 1, xemb.data());
    rm.encode_coordinate(&sy, 1, yemb.data());
    std::vector<float> prompt(a);
    prompt.insert(prompt.end(), xemb.begin(), xemb.end());
    prompt.insert(prompt.end(), yemb.begin(), yemb.end());
    prompt.insert(prompt.end(), b.begin(), b.end());
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int nt = e.step({}, lv.data(), prompt.data(), (int)(prompt.size() / D), hidden.data(), false);
    if (nt == 0) return false;   // eos (plain argmax, per official _detect_gaze)
    // one (x,y) point, include_size=False path
    float xl[1024];
    rm.decode_coordinate(hidden.data(), xl);
    gx = RegionModel::logits_to_center(xl, 1024);
    rm.encode_coordinate(&gx, 1, emb.data());
    e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
    rm.decode_coordinate(hidden.data(), xl);
    gy = RegionModel::logits_to_center(xl, 1024);
    return true;
}
