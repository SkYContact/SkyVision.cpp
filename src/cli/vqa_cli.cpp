// vqa_cli.cpp - unified image/video VQA front end.
//
//   vqa-cli --model <gguf> [--tokenizer <tokenizer.json|hf_dir>] \
//           (--image <file> | --frames <dir> [--fps F]) --ask "question" \
//           [--mode caption|query|detect|point|gaze] [--max-tokens N] \
//           [--temp T] [--top-k K] [--top-p P] [--n-ctx N] [--gpu] \
//           [--vision <gguf>] [--region <gguf>] [--dump-dir D]
//
// The backend is chosen from the GGUF `general.architecture` field. Backends that
// implement vqa::Model run in-process; moondream3.1 (which still carries its own
// richer argument surface) is driven through moondream_main().
#include "vqa.hpp"

#include "json.hpp"
#include "media_io.hpp"
#include "model_registry.hpp"
#include "tokenizer.hpp"
#include "vqa_util.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #include <shellapi.h>
#endif

namespace {

void usage() {
    fprintf(stderr,
        "usage: vqa-cli --model <gguf> [--tokenizer <path|dir>]\n"
        "               [--image <file>] [--video <file> | --frames <dir>] [--fps <f>] [--max-frames <n>]\n"
        "               --ask <question> [--mode <skill>] [options]\n"
        "               --batch <prompts.txt>   text-only batched decode (one prompt per line)\n"
        "               --embed <text>          multimodal embedding (with --image/--video/--audio)\n"
        "               --embed-file <txt>      embed one input per line, JSON lines out\n"
        "               --embed-preset NAME     query|document|sts|classification|clustering|\n"
        "                                       code|qa|fact (default none)\n"
        "               --embed-dim N           Matryoshka truncation (512/256/128/64)\n"
        "               --score <qs.json>       candidate scoring (VideoJev):\n"
        "                                       {\"questions\":[{\"id\",\"task\":\"grounding|choice\",\"query\",\"options\"}]}\n"
        "               [--threshold T]         grounding probability threshold (default: model)\n"
        "               --serve [port]          Web UI (default 8899)\n"
        "               --mcp                   MCP server on stdio (agent tools: see README)\n"
        "\n"
        "options:\n"
        "  --max-tokens N   max new tokens (default 256)\n"
        "  --temp T         sampling temperature, 0 = greedy (default 0)\n"
        "  --top-k K        top-k sampling\n"
        "  --top-p P        nucleus sampling\n"
        "  --n-ctx N        KV context size; when omitted each backend uses its own\n"
        "                   default (moondream3.1: 1024, videochat3: 8192). Larger\n"
        "                   values cost VRAM: moondream needs ~196 KB per token\n"
        "  --threads N      CPU threads\n"
        "  --gpu            prefer CUDA (default when the build has it)\n"
        "  --cpu            force the CPU backend\n"
        "  --vision <gguf>  vision tower file (moondream3.1)\n"
        "  --region <gguf>  region head file (moondream3.1 detect/point/gaze)\n"
        "  --dump-dir D     write intermediate tensors (parity debugging)\n"
        "  --dump-ids       tokenize --ask and print the ids, then exit\n"
        "  --arch NAME      force a backend instead of reading the GGUF field\n"
        "  --quiet          do not stream tokens to stderr\n");
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    // rebuild argv as UTF-8: the default ANSI argv mangles CJK prompts/paths
    {
        int wargc = 0;
        LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargv) {
            static std::vector<std::string> utf8_args;
            static std::vector<char *> utf8_argv;
            for (int i = 0; i < wargc; i++) {
                int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(n - 1, 0);
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
                utf8_args.push_back(std::move(s));
            }
            for (auto & s : utf8_args) utf8_argv.push_back(s.data());
            argv = utf8_argv.data();
            argc = wargc;
            LocalFree(wargv);
        }
    }
#endif

    std::string model_path, tokenizer_arg, image_path, frames_dir, video_path, audio_path, speak_wav, ask_text, mode, arch_override, batch_path,
                dump_dir, vision_path, region_path, score_path, embed_text, embed_file,
                embed_preset = "none";
    bool embed_given = false;
    int max_tokens = 256, n_ctx = 0, top_k = 0, threads = 0;   // 0 = backend default
    int ep_dim = 0;
    float temp = 0.0f, top_p = 1.0f, fps = 0.0f, threshold = 0.0f;
    int max_frames = 32;
    int serve_port = 0;
    std::string serve_host = "127.0.0.1";
    std::string models_dir;          // --models-dir (Web UI / MCP discovery root)
    bool mcp_mode = false;
    bool use_gpu = true, quiet = false, dump_ids = false;   // CUDA preferred, CPU fallback

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--model" || a == "-m") model_path = next("--model");
        else if (a == "--tokenizer" || a == "-t") tokenizer_arg = next("--tokenizer");
        else if (a == "--image" || a == "-i") image_path = next("--image");
        else if (a == "--frames") frames_dir = next("--frames");
        else if (a == "--fps") fps = (float)atof(next("--fps"));
        else if (a == "--video") video_path = next("--video");
        else if (a == "--audio") audio_path = next("--audio");
        else if (a == "--speak") speak_wav = next("--speak");
        else if (a == "--max-frames") max_frames = atoi(next("--max-frames"));
        else if (a == "--serve") { serve_port = atoi(next("--serve")); if (serve_port <= 0) serve_port = 8899; }
        else if (a == "--host") serve_host = next("--host");
        else if (a == "--models-dir") models_dir = next("--models-dir");
        else if (a == "--mcp") mcp_mode = true;
        else if (a == "--batch") batch_path = next("--batch");
        else if (a == "--ask" || a == "-p") ask_text = next("--ask");
        else if (a == "--mode") mode = next("--mode");
        else if (a == "--max-tokens" || a == "-n") max_tokens = atoi(next("--max-tokens"));
        else if (a == "--temp") temp = (float)atof(next("--temp"));
        else if (a == "--top-k") top_k = atoi(next("--top-k"));
        else if (a == "--top-p") top_p = (float)atof(next("--top-p"));
        else if (a == "--n-ctx") n_ctx = atoi(next("--n-ctx"));
        else if (a == "--threads") threads = atoi(next("--threads"));
        else if (a == "--vision") vision_path = next("--vision");
        else if (a == "--region") region_path = next("--region");
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--cpu") use_gpu = false;
        else if (a == "--dump-dir") dump_dir = next("--dump-dir");
        else if (a == "--dump-ids") dump_ids = true;
        else if (a == "--arch") arch_override = next("--arch");
        else if (a == "--score") score_path = next("--score");
        else if (a == "--embed") { embed_text = next("--embed"); embed_given = true; }
        else if (a == "--embed-file") embed_file = next("--embed-file");
        else if (a == "--embed-preset") embed_preset = next("--embed-preset");
        else if (a == "--embed-dim") ep_dim = atoi(next("--embed-dim"));
        else if (a == "--threshold") threshold = (float)atof(next("--threshold"));
        else if (a == "--quiet") quiet = true;
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }
    // --- long-running front ends (no inference flags needed) ---
    if (mcp_mode) {
        // default to ./models when it exists: that is where the weights live
        std::string dir = models_dir;
        if (dir.empty()) dir = vqa::path_exists("models") ? "models" : ".";
        return vqa::run_mcp_server(dir, tokenizer_arg.empty() ? "" : tokenizer_arg, model_path);
    }
    if (serve_port > 0) {
        const std::string dir = models_dir.empty() ? "." : models_dir;
        return vqa::run_web_ui(serve_port, dir, "web", tokenizer_arg.empty() ? "" : tokenizer_arg, serve_host.c_str());
    }

    if (model_path.empty()) { usage(); return 2; }

    std::string arch = arch_override.empty() ? vqa::architecture_of(model_path.c_str()) : arch_override;
    if (arch.empty()) {
        fprintf(stderr, "cannot read general.architecture from %s\n", model_path.c_str());
        return 3;
    }
    fprintf(stderr, "architecture: %s\n", arch.c_str());

    // pure video towers (videomae / vivit) have no text side at all: no tokenizer
    const bool needs_tokenizer = vqa::arch_info(arch).needs_tokenizer;
    std::string tok_path = vqa::resolve_tokenizer_path(tokenizer_arg);
    if (!tok_path.empty()) {
        // keep the explicit argument as given
    } else {
        size_t slash = model_path.find_last_of("/\\");
        std::string dir = slash == std::string::npos ? std::string(".") : model_path.substr(0, slash);
        tok_path = vqa::resolve_tokenizer_path(dir);
    }
    if (tok_path.empty() && needs_tokenizer) {
        // last resort: the arch's canonical checkpoint dir. Never fall back to a
        // tokenizer.json merely sitting in the working directory - for a
        // Qwen-family arch fed the moondream one it fails far from the cause
        // ("placeholder mismatch: 0 image_pad tokens").
        const std::string hint = vqa::resolve_tokenizer_path(vqa::default_tokenizer_dir(arch));
        if (!hint.empty()) {
            tok_path = hint;
            fprintf(stderr, "tokenizer: using %s (pass --tokenizer to override)\n", tok_path.c_str());
        }
    }
    if (tok_path.empty() && needs_tokenizer) {
        fprintf(stderr, "no tokenizer.json found for arch %s (use --tokenizer <dir|tokenizer.json>)\n",
                arch.c_str());
        return 2;
    }

    if (dump_ids) {   // tokenizer self test: no model load needed
        std::vector<int> ids;
        std::string dec;
        if (!vqa::debug_tokenize(tok_path.c_str(), ask_text, ids, dec,
                                 vqa::tokenizer_mode_for_arch(arch))) {
            fprintf(stderr, "tokenizer failed\n");
            return 4;
        }
        printf("ids:");
        for (int id : ids) printf(" %d", id);
        printf("\ndecoded: %s\n", dec.c_str());
        return 0;
    }

    vqa::Model * model = vqa::vqa_create_model(arch);
    if (!model) {
        fprintf(stderr, "architecture '%s' is not wired into this build\n", arch.c_str());
        return 3;
    }
    // video support is a per-backend capability (e.g. moondream3.1 rejects it).
    // caps() is authoritative only after load(), so the pure video towers are
    // let through via ArchInfo (they take frame dirs by definition).
    if (!frames_dir.empty() && !model->caps().video && !vqa::arch_info(arch).pure_vision) {
        fprintf(stderr, "arch %s does not take video input\n", arch.c_str());
        return 2;
    }

    vqa::MediaInput media;
    if (!image_path.empty()) {
        if (!vqa::load_image(image_path, media)) return 4;
    } else if (!video_path.empty()) {
        if (!vqa::load_video_file(video_path, fps, max_frames, media)) return 4;
    } else if (!frames_dir.empty()) {
        if (!vqa::load_frame_dir(frames_dir, fps, media)) return 4;
    }
    if (!audio_path.empty()) {
        if (!vqa::load_audio(audio_path, media)) return 4;
    }
    if (media.empty() && !media.has_audio() && ask_text.empty() && batch_path.empty() &&
        score_path.empty() && !embed_given && embed_file.empty()) { usage(); return 2; }

    vqa::GenParams gp;
    if (!speak_wav.empty()) gp.speak_wav = speak_wav.c_str();
    gp.max_tokens = max_tokens;
    gp.temperature = temp;
    gp.top_k = top_k;
    gp.top_p = top_p;
    gp.stream = !quiet;
    gp.on_token = [](const char * piece, void *) {
        fputs(piece, stderr);
        fflush(stderr);
    };

    vqa::LoadOptions opt;
    opt.n_ctx = n_ctx;
    opt.n_threads = threads;
    opt.use_gpu = use_gpu;
    opt.dump_dir = dump_dir;
    opt.vision_path = vision_path;
    opt.region_path = region_path;
    if (!model->load(model_path, tok_path, opt)) { fprintf(stderr, "model load failed\n"); return 4; }
    if (!batch_path.empty()) {
        const int rc = model->run_batch(batch_path, max_tokens);
        if (rc < 0) { fprintf(stderr, "this backend does not support --batch\n"); return 4; }
        delete model;
        return rc;
    }

    if (!score_path.empty()) {
        // {"questions": [{"id","task":"grounding"|"choice","query","options":[]}]}
        std::string doc_text, err;
        if (!vqa::read_file(score_path.c_str(), doc_text)) {
            fprintf(stderr, "cannot read %s\n", score_path.c_str());
            delete model;
            return 2;
        }
        vqa::json::Value doc;
        if (!vqa::json::parse(doc_text, doc, err)) {
            fprintf(stderr, "%s: %s\n", score_path.c_str(), err.c_str());
            delete model;
            return 2;
        }
        const vqa::json::Value * rows = doc.get("questions");
        if (!rows || !rows->is_array() || rows->items.empty()) {
            fprintf(stderr, "%s: expected {\"questions\":[...]}\n", score_path.c_str());
            delete model;
            return 2;
        }
        std::vector<vqa::ScoreQuery> queries;
        for (size_t i = 0; i < rows->items.size(); i++) {
            const vqa::json::Value & item = rows->items[i];
            vqa::ScoreQuery q;
            q.id = item.str("id", std::to_string(i));
            q.task = item.str("task", "grounding");
            q.query = item.str("query");
            if (const vqa::json::Value * opts = item.get("options")) {
                for (const auto & o : opts->items)
                    if (o.is_string()) q.options.push_back(o.text);
            }
            queries.push_back(std::move(q));
        }
        std::vector<vqa::ScoreResult> results;
        if (!model->score(media, queries, threshold, results, gp)) {
            fprintf(stderr, "scoring failed (this backend/checkpoint has no decision head?)\n");
            delete model;
            return 5;
        }
        vqa::json::Value arr = vqa::json::Value::make_array();
        for (const auto & r : results) {
            vqa::json::Value o = vqa::json::Value::make_object();
            o.set("id", vqa::json::Value::make_string(r.id));
            o.set("task", vqa::json::Value::make_string(r.task));
            vqa::json::Value probs = vqa::json::Value::make_array();
            for (float p : r.probabilities) probs.push(vqa::json::Value::make_number((double)p));
            o.set("probabilities", probs);
            if (r.task == "choice") {
                o.set("answer_index", vqa::json::Value::make_number((double)r.answer_index));
                o.set("answer", vqa::json::Value::make_string(r.answer));
            } else {
                vqa::json::Value times = vqa::json::Value::make_array();
                for (double t : r.times) times.push(vqa::json::Value::make_number(t));
                o.set("times", times);
                o.set("start", vqa::json::Value::make_number(r.start));
                o.set("end", vqa::json::Value::make_number(r.end));
            }
            arr.push(o);
        }
        printf("%s\n", vqa::json::dump(arr).c_str());
        delete model;
        return 0;
    }


    if (embed_given || !embed_file.empty()) {
        vqa::EmbedParams ep;
        ep.preset = embed_preset.c_str();
        ep.truncate_dim = ep_dim;
        if (embed_given) {
            std::vector<float> e;
            if (!model->embed(media, embed_text, ep, e)) {
                fprintf(stderr, "embedding failed (backend has no embed head?)\n");
                delete model; return 5;
            }
            printf("{\"dim\": %d, \"embedding\": [", (int)e.size());
            for (size_t i = 0; i < e.size(); i++) printf("%s%.6f", i ? ", " : "", (double)e[i]);
            printf("]}\n");
        } else {
            std::string doc;
            if (!vqa::read_file(embed_file.c_str(), doc)) {
                fprintf(stderr, "cannot read %s\n", embed_file.c_str()); delete model; return 2;
            }
            size_t idx = 0;
            size_t pos = 0;
            while (pos <= doc.size()) {
                size_t nl = doc.find('\n', pos);
                if (nl == std::string::npos) nl = doc.size();
                std::string line = doc.substr(pos, nl - pos);
                pos = nl + 1;
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                if (line.empty()) continue;
                std::vector<float> e;
                if (!model->embed(media, line, ep, e)) {
                    fprintf(stderr, "embedding failed on line %zu\n", idx + 1);
                    delete model; return 5;
                }
                printf("{\"i\": %zu, \"dim\": %d, \"embedding\": [", idx++, (int)e.size());
                for (size_t i = 0; i < e.size(); i++) printf("%s%.6f", i ? ", " : "", (double)e[i]);
                printf("]}\n");
                fflush(stdout);
            }
        }
        delete model;
        return 0;
    }

    std::string out;
    if (!model->ask(media, ask_text, mode, gp, out)) { fprintf(stderr, "inference failed\n"); return 5; }
    if (!quiet) fputc('\n', stderr);
    printf("%s\n", out.c_str());
    delete model;
    return 0;
}
