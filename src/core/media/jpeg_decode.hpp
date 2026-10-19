// jpeg_decode.hpp - JPEG decoding through libjpeg-turbo (vendored at
// vendor/libjpeg-turbo, built via add_subdirectory). Only compiled when the
// CMake layer defines VQA_HAVE_LIBJPEG; media_io.cpp falls back to stb_image
// otherwise. Rationale: stb's IDCT differs from libjpeg by 1-3 gray levels,
// which broke pixel parity with PIL/libjpeg-based pipelines; libjpeg-turbo is
// the same decoder family as PIL, so RGB output is bit-exact.
//
// Dispatch is by content (SOI magic 0xFF 0xD8), not by file extension, so a
// mislabeled file still goes to the engine that can actually read it.
#pragma once

#include <cstdint>

#include <string>
#include <vector>

namespace vqa {

// First bytes of a JPEG stream (Start Of Image marker).
inline bool looks_like_jpeg(const uint8_t * data, size_t size) {
    return size >= 2 && data[0] == 0xFF && data[1] == 0xD8;
}

// Decode a JPEG held in memory into tightly packed RGB888 (width*height*3).
// Baseline and progressive, any subsampling, grayscale + (best-effort) CMYK.
// On failure returns false and puts libjpeg's message into `err`.
bool jpeg_decode_mem(const uint8_t * data, size_t size,
                     std::vector<uint8_t> & rgb, int & width, int & height,
                     std::string & err);

// Same, reading the file into memory first.
bool jpeg_decode_file(const std::string & path,
                      std::vector<uint8_t> & rgb, int & width, int & height,
                      std::string & err);

}  // namespace vqa
