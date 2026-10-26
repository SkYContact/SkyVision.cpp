// vqa.hpp - unified multimodal VQA model interface.
//
// The engine is a multi-model image/video VQA runtime: every architecture
// (moondream3.1, VideoChat3, ...) is a backend implementing vqa::Model, and the
// CLI / server only talk to this interface plus the capability table.
//
// MediaInput carries a frame sequence: a single frame for images, N sampled
// frames plus timestamps for video. Backends that do not support video report
// it through Capabilities and reject kind == Video.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vqa {

enum class MediaKind { Image, Video };

struct Frame {
    std::vector<uint8_t> rgb;   // interleaved RGB8, width*height*3
};

struct MediaInput {
    MediaKind kind = MediaKind::Image;
    int width = 0, height = 0;                       // all frames share this size
    std::vector<Frame> frames;                       // 1 for images, N for video
    double fps = 0.0;                                // video only (0 if unknown)
    std::vector<double> timestamps;                  // seconds, one per frame (video)
    // optional audio track: mono float PCM at sample_rate (16 kHz when it comes
    // from load_audio). Qwen2.5-Omni style - a model consumes either frames,
    // audio, or both.
    std::vector<float> audio;
    int sample_rate = 0;

    bool empty() const { return frames.empty(); }
    bool has_audio() const { return !audio.empty() && sample_rate > 0; }
};

struct GenParams {
    int max_tokens = 256;
    float temperature = 0.0f;    // 0 == greedy
    int top_k = 0;               // 0 == disabled
    float top_p = 1.0f;
    int seed = -1;
    bool stream = true;
    void (*on_token)(const char * piece, void * ud) = nullptr;   // optional streaming sink
    void * on_token_ud = nullptr;
    // optional coarse progress (e.g. per vision clip: "vision", done, total)
    void (*on_stage)(const char * stage, int done, int total, void * ud) = nullptr;
    void * on_stage_ud = nullptr;
    // multi-turn conversation prefix: fully formatted prior turns (with chat
    // markers), prepended verbatim before the new user turn. Backends without
    // multi-turn support ignore it.
    std::string history;
    // speech output (qwen2.5-omni talker): write 24 kHz wav here when set.
    // Backends without a talker ignore it.
    const char * speak_wav = nullptr;
    std::string speaker;          // talker voice name (qwen3-omni: chelsie/ethan/aiden)
    int max_codes = 400;          // talker codec-frame cap (anti-runaway on long text)
};

// Candidate scoring: a "System One" video model does not decode an answer, it
// scores a fixed candidate list in one forward pass (temporal grounding: one
// candidate per video segment; multiple choice: one candidate per option).
// Backends without a decision head return false from Model::score().
struct ScoreQuery {
    std::string id;
    std::string task = "grounding";   // "grounding" | "choice"
    std::string query;
    std::vector<std::string> options; // choice only
};

struct ScoreResult {
    std::string id, task;
    std::vector<float> probabilities;   // one per candidate
    std::vector<double> times;          // grounding: candidate timestamps (seconds)
    int answer_index = -1;              // choice only (0-based)
    std::string answer;                 // choice only
    double start = -1.0, end = -1.0;    // grounding: decoded interval (seconds)
};

// Embedding request knobs (EmbeddingGemma2-style encoders).
//   preset       task-instruction prefix prepended to the text ("none" = raw);
//                the official presets are the sentence-transformers prompts:
//                query / document / sts / classification / clustering / qa /
//                fact / code. Media-only requests ignore it.
//   truncate_dim Matryoshka output dimension (0 = the model's default, 768);
//                the slice is re-normalized, matching sentence-transformers'
//                truncate_dim.
struct EmbedParams {
    const char * preset = "none";
    int truncate_dim = 0;
};

// Load-time configuration shared by every backend.
struct LoadOptions {
    int n_ctx = 8192;      // KV context window
    int n_threads = 0;     // 0 = auto (CPU backend)
    bool use_gpu = true;   // prefer the CUDA backend; silently falls back to CPU
                           // when the build has no GGML_CUDA (same convention as
                           // the moondream engine, where only MD_CPU forces CPU)
    std::string dump_dir;  // write intermediate tensors here (parity debugging)
    std::string vision_path, region_path;   // moondream3 side models (gguf siblings)
};

struct Capabilities {
    bool image = true;
    bool video = false;
    bool audio = false;          // consumes audio tracks (omni / embgemma2)
    // moondream3.1 skill surface
    bool caption = false;        // short / normal / long
    bool detect = false;         // bounding boxes
    bool point = false;          // points
    bool gaze = false;           // gaze target from a start point
    // VideoChat3 style streaming/proactive surface
    bool proactive_states = false;   // emits </Silence> </Standby> </Response>
    bool timestamps_prompt = false;  // prompt carries <N.N seconds> frame markers
};

// A loaded model instance. One instance owns its weights + KV cache and is not
// thread safe; callers use one instance per concurrent request.
class Model {
public:
    virtual ~Model() = default;

    // gguf_path: model file. tokenizer_path: tokenizer.json (or an HF model dir,
    // in which case "tokenizer.json" is appended).
    virtual bool load(const std::string & gguf_path, const std::string & tokenizer_path,
                      const LoadOptions & opt = LoadOptions()) = 0;

    virtual const Capabilities & caps() const = 0;
    virtual const char * name() const = 0;

    // Single-turn VQA. `prompt` is the raw user question ("Describe the image.").
    // `mode` is an optional skill selector for backends that have one
    // (moondream: "caption" | "query" | "detect" | "point" | "gaze"; others ignore it).
    virtual bool ask(const MediaInput & media, const std::string & prompt,
                     const std::string & mode, const GenParams & gp,
                     std::string & out_text) = 0;

    // Realtime streaming session: like ask(), but turn 1 establishes a live
    // session and later turns append frames/prompts incrementally onto the
    // KV caches (no full-context rebuild). Backends without session support
    // fall back to plain ask(). session_reset() drops the live state.
    virtual bool ask_session(const MediaInput & media, const std::string & prompt,
                             const GenParams & gp, std::string & out_text) {
        return ask(media, prompt, "", gp, out_text);
    }
    virtual void session_reset() {}

    // Candidate scoring (VideoJev style decision head). `threshold` is the
    // grounding decoder's probability threshold (<= 0 keeps the model default).
    // Backends without a decision head leave this returning false.
    virtual bool score(const MediaInput & media, const std::vector<ScoreQuery> & queries,
                       float threshold, std::vector<ScoreResult> & out,
                       const GenParams & gp = GenParams()) {
        (void)media; (void)queries; (void)threshold; (void)out; (void)gp;
        return false;
    }

    // batched text-only decode driver (one prompt per line); -1 = unsupported
    virtual int run_batch(const std::string & batch_path, int n_gen) { (void)batch_path; (void)n_gen; return -1; }

    // Multimodal embedding: `media` may carry frames and/or audio, `text` may be
    // empty; at least one of them must be set. The output is one L2-normalized
    // vector of truncate_dim (0 = model default) floats. Returns false when the
    // backend has no embedding head.
    virtual bool embed(const MediaInput & media, const std::string & text,
                       const EmbedParams & ep, std::vector<float> & out) {
        (void)media; (void)text; (void)ep; (void)out;
        return false;
    }

    // Chat-template knowledge the front ends need but only the backend has.
    // format_history_turn renders one prior turn for gp.history ("" = this
    // backend has no multi-turn template; front ends then skip the prefix).
    // format_system renders a system prompt prelude for a first realtime turn
    // ("" = none). silence_piece is the token text meaning "nothing to say"
    // in realtime narration ("" = no such token).
    virtual std::string format_history_turn(const std::string & user,
                                            const std::string & assistant) const {
        (void)user; (void)assistant; return std::string();
    }
    virtual std::string format_system(const std::string & sys) const { (void)sys; return std::string(); }
    virtual const char * silence_piece() const { return ""; }
    // Default realtime-narration system prompt for this architecture. The
    // protocol it describes (three-state markers, or <|silence|>) is model
    // specific, so the backend owns the text; "" = let the front end decide.
    virtual const char * realtime_system_prompt() const { return ""; }
};

// Backend factories (one per architecture) live in backend_factories.hpp,
// included only by vqa_registry.cpp - wiring a new backend touches that header
// plus one table row in vqa_registry.cpp, and nothing else in core.

// Resolve a backend by GGUF architecture name; returns nullptr when unsupported.
Model * vqa_create_model(const std::string & arch);

// What a front end needs to know about an architecture before loading it: can
// this build run it, and which files must sit next to the GGUF. Discovery uses
// this instead of keeping its own copy of the backend list.
struct ArchInfo {
    bool supported = false;
    bool needs_tokenizer = true;      // tokenizer.json (or an HF checkpoint dir)
    bool needs_vision_gguf = false;   // moondream3: a *-vision.gguf sibling
    bool pure_vision = false;         // video tower: no tokenizer, no text side
    const char * web_page = "vc3.html";  // generic web UI page for this arch
};

ArchInfo arch_info(const std::string & arch);

// unified Web UI (vqa-cli --serve); returns process exit code.
// `use_gpu` carries the CLI's --cpu/--gpu choice into on-demand loads (it used
// to be swallowed, so a "--cpu" server still loaded weights onto the GPU).
int run_web_ui(int port, const std::string & models_dir, const std::string & web_dir,
               const std::string & tokenizer, const char * bind_host,
               bool use_gpu = true);

// MCP (Model Context Protocol) server over stdio (vqa-cli --mcp): exposes the
// engine as agent-facing tools. `preset_model` is an optional id / gguf path
// used when a tool call omits its "model" argument. Returns process exit code.
int run_mcp_server(const std::string & models_dir, const std::string & tokenizer,
                   const std::string & preset_model);

// Peek "general.architecture" in a GGUF file without loading tensors.
// Returns "" when the file cannot be read.
std::string architecture_of(const char * gguf_path);

// Resolve a tokenizer argument: a path to tokenizer.json, or a directory
// containing one. Returns the resolved file path (unchanged when it exists as a
// file, empty when nothing was found).
std::string resolve_tokenizer_path(const std::string & arg);

// Fallback vocab size when a GGUF carries no "<arch>.vocab_size": used by
// discovery to match a tokenizer.json against the model family.
size_t fallback_vocab_target(const std::string & arch);

// Repo-layout default checkpoint dir whose tokenizer.json serves `arch`
// ("models/VideoChat3-4B" etc.). Empty when the arch has no canonical dir or
// needs no tokenizer (videomae/vivit). Callers must still check existence - a
// hint, never a guarantee.
std::string default_tokenizer_dir(const std::string & arch);

}  // namespace vqa
