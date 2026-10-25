// vqa_spec.cpp - the spec table: one row per GGUF architecture label.
//
// Same discipline as the kArchs table in vqa_registry.cpp: this is the single
// place that knows what each backend can do, so wiring a new backend touches
// that table, the factory header, and one row here. Capability truth was
// transcribed from each backend's caps()/virtual overrides; when a backend
// gains an ability, both places move together.
#include "core/runtime/vqa_spec.hpp"

#include <cstring>

namespace vqa {

const char * task_name(Task t) {
    switch (t) {
        case Task::Ask:      return "ask";
        case Task::Caption:  return "caption";
        case Task::Detect:   return "detect";
        case Task::Point:    return "point";
        case Task::Gaze:     return "gaze";
        case Task::Classify: return "classify";
        case Task::Embed:    return "embed";
        case Task::Score:    return "score";
        case Task::Gate:     return "gate";
    }
    return "";
}

bool task_from_name(const char * name, Task & out) {
    static const struct { const char * n; Task t; } k[] = {
        { "ask", Task::Ask },          { "caption", Task::Caption },
        { "detect", Task::Detect },    { "point", Task::Point },
        { "gaze", Task::Gaze },        { "classify", Task::Classify },
        { "embed", Task::Embed },      { "score", Task::Score },
        { "gate", Task::Gate },
    };
    if (!name) return false;
    for (const auto & e : k)
        if (std::strcmp(name, e.n) == 0) { out = e.t; return true; }
    return false;
}

const char * mode_name(Mode m) {
    return m == Mode::Streaming ? "streaming" : "offline";
}

bool mode_from_name(const char * name, Mode & out) {
    if (!name) return false;
    if (std::strcmp(name, "offline") == 0) { out = Mode::Offline; return true; }
    if (std::strcmp(name, "streaming") == 0) { out = Mode::Streaming; return true; }
    return false;
}

namespace {

struct SpecRow {
    const char * canonical;
    const char * aliases;        // space-separated ("" = none)
    const char * inputs;         // space-separated subset of image video audio text
    const char * tasks;          // space-separated task tokens
    bool streaming;
    bool pure_vision;
    bool needs_vision_gguf;
    bool speech_output;
    const char * default_dir;
    const char * web_page;       // "" = generic vc3.html
    const char * note;
};

const SpecRow kSpecs[] = {
    { "moondream3", "moondream", "image text", "ask caption detect point gaze",
      false, false, true, false, "models/moondream3.1", "moondream.html",
      "caption needs the *-vision.gguf sibling; detect/point/gaze need the *-region.gguf sibling" },
    { "videochat3", "", "image video text", "ask",
      false, false, false, false, "models/VideoChat3-4B", "",
      "proactive states (</Silence> </Standby> </Response>) in realtime prompts" },
    { "qwen3_5", "", "image video text", "ask caption",
      false, false, false, false, "models/Qwen3.5-2B", "",
      "caption = the Marlin-2B trained skills (--mode caption/find)" },
    { "marlin", "", "image video text", "ask caption",
      false, false, false, false, "models/Marlin-2B", "",
      "caption/find carry the official trained prompts (--mode caption / --mode find)" },
    { "videojev", "", "video text", "score",
      false, false, false, false, "models/VideoJev-4B", "",
      "score-only decision head: no text is generated (--score qs.json)" },
    { "minicpmv47", "minicpm_v4_7", "image video text", "ask",
      false, false, false, false, "models/MiniCPM-V-4.7-1B", "",
      "" },
    { "qwen3vl", "qwen3_vl", "image video text", "ask",
      true, false, false, false, "models/OneStreamer-4B", "",
      "OVO three-state streaming (</Silence> </Standby> </Response>) with PHCM session memory" },
    { "moss_vl", "", "image video text", "ask",
      true, false, false, false, "models/Moss-VL-Realtime-q4k", "moss.html",
      "cross-attention realtime: <|silence|> marks an idle round; rt/start keeps a resident stream" },
    { "smolvlm", "idefics3", "image text", "ask",
      false, false, false, false, "", "", "" },
    { "qwen2vl", "qwen2_5_vl", "image text", "ask",
      false, false, false, false, "", "",
      "video input not wired yet (image only)" },
    { "videomae", "vivit", "image video", "classify embed",
      false, true, false, false, "", "",
      "classifier needs a *-finetuned-* checkpoint; the plain release embeds only. vivit = same graph, cls token + learned positions" },
    { "vjepa2", "vjepa", "image video", "classify embed",
      false, true, false, false, "", "",
      "three heads by checkpoint: encoder-only embeds; SSv2 finetune classifies (--mode N); --mode predict runs the masked-latent predictor" },
    { "mage_vl", "", "image video text", "ask gate",
      true, false, false, false, "models/Mage-VL", "",
      "gate = system 1 speak/no-speak head (--mode gate / --mode gated), auto-applied on realtime observe rounds" },
    { "qwen25omni", "qwen2_5_omni", "image audio text", "ask",
      false, false, false, true, "models/Qwen2.5-Omni-3B", "omni.html",
      "video frames not wired yet; speech output needs the talker gguf sibling (--speak)" },
    { "embgemma2", "", "image video audio text", "embed",
      false, false, false, false, "models/embeddinggemma-2", "",
      "768-d Matryoshka vectors, truncate to 512/256/128/64 (--embed-dim)" },
    { "gemma4", "", "image video audio text", "ask",
      false, false, false, false, "models/gemma-4-12B", "",
      "encoder-free: linear vision (48px patches, <=280 soft tokens) + raw-waveform audio (640 samples/token); video = HF official framing (70-token budget, per-frame mm:ss blocks)" },
    { "siglip", "", "image", "classify embed",
      false, true, false, false, "models/siglip-base", "",
      "classifier needs a classification-head checkpoint (e.g. NSFW); the base release embeds only" },
    { "keye-vl2", "keye_vl2", "image video text", "ask",
      false, false, false, false, "models/keye-vl2", "",
      "expert residency cache (KEYE_EXPERTCACHE=1) fits the 19GB checkpoint a 12GB card. "
      "VIDEO IS A SIMPLIFIED PER-FRAME PATH, not the official slow/fast dual-stream "
      "packing pipeline (official: packed sequence through the tower with "
      "packing_position_embedding + cross-frame varlen attention + <|frame|><hh:mm:ss> "
      "markers + fast stream <|fast_start|>..|>); porting the official path is tracked as a TODO" },
    { "janus", "janus_pro", "image text", "ask",
      false, false, false, false, "models/Janus-Pro-1B", "",
      "understanding path only; the generation VQ decoder is not part of this engine" },
};

ArchSpec row_to_spec(const SpecRow & r) {
    ArchSpec s;
    s.canonical = r.canonical;
    const char * p = r.aliases;
    while (p && *p) {
        const char * sp = std::strchr(p, ' ');
        size_t n = sp ? size_t(sp - p) : std::strlen(p);
        if (n) s.aliases.emplace_back(p, n);
        p = sp ? sp + 1 : nullptr;
    }
    auto split = [](const char * csv, auto push) {
        const char * q = csv;
        while (q && *q) {
            const char * sp = std::strchr(q, ' ');
            size_t n = sp ? size_t(sp - q) : std::strlen(q);
            if (n) push(std::string(q, n));
            q = sp ? sp + 1 : nullptr;
        }
    };
    split(r.inputs, [&](std::string v) { s.inputs.push_back(std::move(v)); });
    split(r.tasks, [&](std::string v) {
        Task t;
        if (task_from_name(v.c_str(), t)) s.tasks.push_back(t);
    });
    s.modes.push_back(Mode::Offline);
    if (r.streaming) s.modes.push_back(Mode::Streaming);
    s.pure_vision = r.pure_vision;
    s.needs_vision_gguf = r.needs_vision_gguf;
    s.speech_output = r.speech_output;
    s.default_dir = r.default_dir;
    s.web_page = r.web_page;
    s.note = r.note;
    return s;
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void append_json_array(std::string & out, const std::vector<std::string> & v) {
    out += "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) out += ",";
        out += "\"" + json_escape(v[i]) + "\"";
    }
    out += "]";
}

}  // namespace

const ArchSpec * spec_of(const std::string & arch_label) {
    static const std::vector<ArchSpec> kAll = [] {
        std::vector<ArchSpec> v;
        for (const auto & r : kSpecs) v.push_back(row_to_spec(r));
        return v;
    }();
    for (const auto & s : kAll) {
        if (s.canonical == arch_label) return &s;
        for (const auto & a : s.aliases)
            if (a == arch_label) return &s;
    }
    return nullptr;
}

std::vector<ArchSpec> all_specs() {
    std::vector<ArchSpec> v;
    for (const auto & r : kSpecs) v.push_back(row_to_spec(r));
    return v;
}

bool supports(const ArchSpec & spec, Task task, Mode mode) {
    for (Task t : spec.tasks)
        if (t == task) {
            if (mode == Mode::Offline) return true;
            for (Mode m : spec.modes)
                if (m == Mode::Streaming) return true;
            return false;
        }
    return false;
}

std::string specs_to_json() {
    std::string out = "[";
    bool first = true;
    for (const auto & s : all_specs()) {
        if (!first) out += ",";
        first = false;
        out += "{\"arch\":\"" + json_escape(s.canonical) + "\",";
        out += "\"aliases\":";
        std::vector<std::string> alias_names(s.aliases);
        append_json_array(out, alias_names);
        out += ",\"inputs\":";
        append_json_array(out, s.inputs);
        out += ",\"tasks\":[";
        for (size_t i = 0; i < s.tasks.size(); i++) {
            if (i) out += ",";
            out += "\"" + std::string(task_name(s.tasks[i])) + "\"";
        }
        out += "],\"modes\":[";
        for (size_t i = 0; i < s.modes.size(); i++) {
            if (i) out += ",";
            out += "\"" + std::string(mode_name(s.modes[i])) + "\"";
        }
        out += "]";
        out += std::string(",\"pure_vision\":") + (s.pure_vision ? "true" : "false");
        out += std::string(",\"needs_vision_gguf\":") + (s.needs_vision_gguf ? "true" : "false");
        out += std::string(",\"speech_output\":") + (s.speech_output ? "true" : "false");
        out += ",\"default_dir\":\"" + json_escape(s.default_dir) + "\"";
        out += ",\"web_page\":\"" + json_escape(s.web_page) + "\"";
        out += ",\"note\":\"" + json_escape(s.note) + "\"";
        out += "}";
    }
    out += "]";
    return out;
}

}  // namespace vqa
