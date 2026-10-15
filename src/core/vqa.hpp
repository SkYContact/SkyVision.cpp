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

    bool empty() const { return frames.empty(); }
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

    // batched text-only decode driver (one prompt per line); -1 = unsupported
    virtual int run_batch(const std::string & batch_path, int n_gen) { (void)batch_path; (void)n_gen; return -1; }
};

// Backend factories (one per architecture).
Model * create_moondream();
Model * create_videochat3();

// Resolve a backend by GGUF architecture name; returns nullptr when unsupported.
Model * vqa_create_model(const std::string & arch);

// unified Web UI (vqa-cli --serve); returns process exit code
int run_web_ui(int port, const std::string & models_dir, const std::string & web_dir,
               const std::string & tokenizer, const char * bind_host);

// Peek "general.architecture" in a GGUF file without loading tensors.
// Returns "" when the file cannot be read.
std::string architecture_of(const char * gguf_path);

// Resolve a tokenizer argument: a path to tokenizer.json, or a directory
// containing one. Returns the resolved file path (unchanged when it exists as a
// file, empty when nothing was found).
std::string resolve_tokenizer_path(const std::string & arg);

}  // namespace vqa
