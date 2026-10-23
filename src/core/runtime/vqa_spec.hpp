// vqa_spec.hpp - the capability vocabulary front ends ask about.
//
// The 14 backends have wildly different skills (free VQA, detection, scoring,
// embedding, realtime gating...), but a caller that wants to compose backends
// - the MCP host, the Web UI picker, a script chaining embed -> retrieve ->
// ask - must never need per-backend knowledge. Like audio.cpp's task table,
// the answer to "can this model do X" is always shaped the same: one task
// token from a fixed, build-time vocabulary plus one mode. Anything that
// cannot be phrased in those terms does not get surfaced here (it is a CLI
// flag, not a capability).
//
// Static truth only: a spec describes what an architecture can do with the
// right weights present. Runtime facts that depend on which sibling GGUFs were
// actually found (moondream caption needs the vision gguf, detect needs the
// region gguf) stay in Model::caps() after load - the two layers answer
// different questions and both are kept.
#pragma once

#include <string>
#include <vector>

namespace vqa {

// The task vocabulary this build knows, askable without a model loaded.
enum class Task {
    Ask,        // free-form VQA over media (--ask)
    Caption,    // descriptive captioning skill (moondream caption; marlin --mode caption)
    Detect,     // bounding boxes (moondream region head)
    Point,      // object center points (moondream region head)
    Gaze,       // gaze target from a start point (moondream region head)
    Classify,   // fixed label-set classification head (videomae / vjepa2 / siglip)
    Embed,      // pooled feature vector (siglip / videomae / vjepa2 / embgemma2)
    Score,      // candidate-list scoring decision head (videojev)
    Gate,       // realtime speak / no-speak gate (mage system 1)
};

// Streaming = a live incremental session (Model::ask_session, rt endpoints):
// frames append onto the KV caches instead of a full-context rebuild per turn.
enum class Mode { Offline, Streaming };

const char * task_name(Task t);                        // canonical token, static
bool task_from_name(const char * name, Task & out);    // false on unknown spelling
const char * mode_name(Mode m);
bool mode_from_name(const char * name, Mode & out);

struct ArchSpec {
    std::string canonical;              // canonical GGUF architecture label
    std::vector<std::string> aliases;   // legacy / alternate labels (same spec)
    std::vector<std::string> inputs;    // subset of "image" "video" "audio" "text"
    std::vector<Task> tasks;
    std::vector<Mode> modes;            // Streaming listed only where real
    bool pure_vision = false;           // video tower: no tokenizer, no text side
    bool needs_vision_gguf = false;     // a *-vision.gguf sibling is required
    bool speech_output = false;         // can synthesize speech (omni talker)
    std::string default_dir;            // canonical models/<checkpoint> ("" = none)
    std::string web_page;               // dedicated Web UI page ("" = generic)
    std::string note;                   // condition a caller should surface
};

// Spec for one GGUF architecture label - canonical or alias. Aliases that name
// a distinct model on the same backend (marlin / videojev under qwen3_5) get
// their own spec, because their task sets genuinely differ. nullptr if unknown.
const ArchSpec * spec_of(const std::string & arch_label);
std::vector<ArchSpec> all_specs();

// supports() is the one capability question. Unknown task/mode spellings
// answer false - validate names through task_name()/mode_name() first when the
// distinction between "no" and "misspelled" matters (audio.cpp lesson).
bool supports(const ArchSpec & spec, Task task, Mode mode);

// Machine-readable dump of every spec, one JSON array. Consumed by
// `vqa-cli --specs`, GET /api/specs, and the MCP list_models "tasks" field.
std::string specs_to_json();

}  // namespace vqa
