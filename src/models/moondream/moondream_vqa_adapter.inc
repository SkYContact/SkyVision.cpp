// moondream_vqa_adapter.inc - vqa::Model adapter for the moondream3.1 engine.
// Included from moondream.cpp AFTER Engine/VisionModel/RegionModel and AFTER
// moondream_server.inc (whose static helpers - caption templates, detect/point/
// gaze, image prefill with KV-prefix caching - are reused verbatim), so the
// unified front end (vqa-cli, Web UI) can drive this backend like any other.

namespace {

struct MoondreamModel : vqa::Model {
    Engine e;
    VisionModel vm;
    RegionModel rm;
    bool vm_ok = false, rm_ok = false;

    ~MoondreamModel() override = default;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override {
        if (!e.init(gguf_path.c_str(), tokenizer_path.c_str(), opt.n_ctx > 0 ? std::min(opt.n_ctx, 2048) : 1024))
            return false;
        const char * vp = opt.vision_path.empty() ? "moondream3.1-vision.gguf" : opt.vision_path.c_str();
        vm_ok = vm.load(vp, e.backend);
        if (!vm_ok) {
            ggml_backend_t cpu_be = ggml_backend_cpu_init();
            vm_ok = vm.load(vp, cpu_be);
        }
        if (!vm_ok) fprintf(stderr, "moondream adapter: vision model failed to load\n");
        const char * rp = opt.region_path.empty() ? "moondream3.1-region.gguf" : opt.region_path.c_str();
        rm_ok = rm.load(rp);
        caps_.image = true;
        caps_.video = false;
        return true;
    }

    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "moondream3.1"; }

    // text-only batched decoding (moondream_batch.inc); the unified CLI drives
    // moondream through its own entry point, this keeps the adapter usable too
    int run_batch(const std::string & path, int n_gen) override {
        return ::run_batch(e, path.c_str(), e.n_ctx, n_gen);
    }

    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override {
        if (!vm_ok && !media.empty()) { out_text = "vision model unavailable"; return false; }
        const bool have_image = !media.empty();
        const uint8_t * rgb = have_image ? media.frames[0].rgb.data() : nullptr;
        const int iw = media.width, ih = media.height;
        if (have_image) {
            if ((int)media.frames[0].rgb.size() != (size_t)iw * ih * 3) { out_text = "bad image frame"; return false; }
            md_ensure_img_prefill(e, vm, rgb, iw, ih);
        }

        if (mode == "detect") {
            if (!have_image || !rm_ok) { out_text = "detect needs an image + region model"; return false; }
            auto boxes = md_detect(e, vm, rm, rgb, iw, ih, prompt);
            out_text = "boxes:";
            char b[128];
            for (auto & bx : boxes) {
                snprintf(b, sizeof(b), " %.4f %.4f %.4f %.4f", bx[0], bx[1], bx[2], bx[3]);
                out_text += b;
            }
            return true;
        }
        if (mode == "point") {
            if (!have_image || !rm_ok) { out_text = "point needs an image + region model"; return false; }
            auto pts = md_point(e, vm, rm, rgb, iw, ih, prompt);
            out_text = "points:";
            char b[64];
            for (auto & p : pts) {
                snprintf(b, sizeof(b), " %.4f %.4f", p[0], p[1]);
                out_text += b;
            }
            return true;
        }
        if (mode == "gaze") {
            if (!have_image || !rm_ok) { out_text = "gaze needs an image + region model"; return false; }
            float sx = 0, sy = 0, gx = 0, gy = 0;
            sscanf(prompt.c_str(), "%f,%f", &sx, &sy);
            if (!md_gaze(e, vm, rm, rgb, iw, ih, sx, sy, gx, gy)) { out_text = "no gaze point"; return true; }
            char b[64];
            snprintf(b, sizeof(b), "%.4f %.4f", gx, gy);
            out_text = std::string("gaze: ") + b;
            return true;
        }
        if (mode == "caption") {
            if (!have_image) { out_text = "caption needs an image"; return false; }
            const int t4 = prompt == "short" ? 12492 : (prompt == "long" ? 4059 : 6382);
            std::vector<int> ctoks = { 1, 32708, 2, t4, 3 };
            std::vector<int> out;
            int cur = e.step(ctoks, nullptr);
            out.push_back(cur);
            if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
            for (int i = 1; i < std::min(gp.max_tokens, 512); i++) {
                cur = e.step({ cur }, nullptr);
                if (cur == 0) break;
                out.push_back(cur);
                if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
            }
            out_text = e.tok.decode(out);
            return true;
        }

        // plain ask / query
        std::vector<int> toks = have_image
            ? std::vector<int>{ 1, 15381, 2 }      // bos+image already prefilled
            : std::vector<int>{ 0, 1, 15381, 2 };
        auto q = e.tok.encode(prompt);
        toks.insert(toks.end(), q.begin(), q.end());
        toks.push_back(3);

        // official query() reasoning mode (ref_moondream.py _generate_reasoning):
        // prompt = query prefix + question + thinking_id(4); generate until
        // answer_id(3) — coordinate tokens (5) are region-model grounded, eos_id
        // (0) and size_id (6) logits masked — then feed the query suffix (3) and
        // produce the answer
        if (mode == "ask-think" && rm_ok) {
            const int D = e.model.n_embd;
            std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
            std::vector<int> rtoks = toks;
            rtoks.back() = 4;                      // thinking_id instead of suffix
            int cur = e.step(rtoks, lv.data(), nullptr, 0, hidden.data(), false);
            const int think_cap = 4 * gp.max_tokens + 256;
            for (int i = 0; i < think_cap && cur != 3; i++) {
                if (cur == 5) {                     // coord: region encode/decode
                    float xl[1024];
                    rm.decode_coordinate(hidden.data(), xl);
                    float c = RegionModel::logits_to_center(xl, 1024);
                    rm.encode_coordinate(&c, 1, emb.data());
                    e.step({}, lv.data(), emb.data(), 1, hidden.data(), false);
                } else {
                    if (gp.on_token && cur != 6 && cur != 7 && cur != 9) {
                        std::string piece = e.tok.decode({ cur });
                        // grounding markers can still decode to their reserved
                        // spelling — never surface those
                        if (piece.find("<|md_reserved") == std::string::npos)
                            gp.on_token(piece.c_str(), gp.on_token_ud);
                    }
                    e.step({ cur }, lv.data(), nullptr, 0, hidden.data(), false);
                }
                lv[0] = -INFINITY;                  // mask eos_id and size_id
                lv[6] = -INFINITY;
                cur = (int)(std::max_element(lv.begin(), lv.end()) - lv.begin());
            }
            if (gp.on_token) gp.on_token("\n\n———\n\n", gp.on_token_ud);
            // answer: the query suffix token the reasoning stopped on, then decode
            std::vector<int> out;
            cur = e.step({ 3 }, nullptr);
            if (cur != 0) {
                out.push_back(cur);
                if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
            }
            for (int i = 1; i < gp.max_tokens && cur != 0; i++) {
                cur = e.step({ cur }, nullptr);
                if (cur == 0) break;
                out.push_back(cur);
                if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
            }
            out_text = e.tok.decode(out);
            return true;
        }

        // md_ensure_img_prefill left n_past at 730 (or 0 without an image)
        std::vector<int> out;
        int cur = e.step(toks, nullptr);
        out.push_back(cur);
        if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
        for (int i = 1; i < gp.max_tokens; i++) {
            cur = e.step({ cur }, nullptr);
            if (cur == 0) break;
            out.push_back(cur);
            if (gp.on_token) gp.on_token(e.tok.decode({ cur }).c_str(), gp.on_token_ud);
        }
        out_text = e.tok.decode(out);
        return true;
    }

    vqa::Capabilities caps_;
};

}  // namespace

namespace vqa {
Model * create_moondream() { return new MoondreamModel(); }
}  // namespace vqa
