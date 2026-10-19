// media_io.hpp - shared media decoding for every backend.
//
// Images are decoded with stb_image; videos are taken from a directory of frames
// (sorted by name) since the engine deliberately carries no video codec. The
// result is always a vqa::MediaInput, so a backend only ever sees frames + time.
#pragma once

#include "core/vqa.hpp"

#include <string>

namespace vqa {

// Decode one image file (png/jpg/bmp) into a single-frame MediaInput.
bool load_image(const std::string & path, MediaInput & out);

// Decode a video file via an ffmpeg pipe, sampling n = min(duration*fps,
// max_frames) frames (uniform resample when the cap binds; qwen_vl_utils style).
// Decode a directory of image files as a video. Frames are ordered by filename;
// `fps` (<= 0 -> 1.0) only feeds the timestamps of the prompt.
// Decode an in-memory image buffer (png/jpg/bmp) into a single-frame MediaInput.
bool load_image_from_memory(const std::string & bytes, MediaInput & out);

bool load_frame_dir(const std::string & dir, double fps, MediaInput & out);

// ffmpeg-pipe video decode (uses FFMPEG or ffmpeg from PATH)
bool load_video_file(const std::string & path, double fps, int max_frames, MediaInput & out);

// Decode any ffmpeg-readable audio file (wav/mp3/flac/m4a/ogg) to mono 16 kHz
// float PCM in MediaInput::audio (video_pipe.cpp owns the ffmpeg pipe).
bool load_audio(const std::string & path, MediaInput & out);

}  // namespace vqa
