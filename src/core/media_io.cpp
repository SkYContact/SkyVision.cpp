#include "media_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// the single translation unit that owns the stb_image implementation
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

// JPEG goes through libjpeg-turbo when it is vendored (bit-exact with PIL /
// the libjpeg family); PNG/BMP stay on stb. Dispatch is by content (SOI
// magic), so file extensions never decide the engine.
#ifdef VQA_HAVE_LIBJPEG
#include "jpeg_decode.hpp"
#endif

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace vqa {

namespace {

// JPEG stream sniff: Start Of Image marker.
bool has_soi(const uint8_t * b, size_t n) {
    return looks_like_jpeg(b, n);
}

bool file_has_soi(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t magic[2] = {0, 0};
    size_t n = fread(magic, 1, 2, f);
    fclose(f);
    return has_soi(magic, n);
}

void set_single_frame(MediaInput & out, int w, int h, std::vector<uint8_t> && rgb) {
    out.kind = MediaKind::Image;
    out.width = w;
    out.height = h;
    out.fps = 0.0;
    out.timestamps.clear();
    out.frames.clear();
    Frame f;
    f.rgb = std::move(rgb);
    out.frames.push_back(std::move(f));
}

// production dispatch for one image file (shared by load_image/load_frame_dir)
bool decode_file(const std::string & path, int & w, int & h, std::vector<uint8_t> & rgb) {
#ifdef VQA_HAVE_LIBJPEG
    if (file_has_soi(path)) {
        std::string err;
        if (jpeg_decode_file(path, rgb, w, h, err)) return true;
        // keep the historical (stb) behavior alive on libjpeg failures instead
        // of failing inputs that used to work
        fprintf(stderr, "libjpeg decode failed (%s), falling back to stb: %s\n", err.c_str(), path.c_str());
    }
#endif
    int comp = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!data) {
        fprintf(stderr, "cannot decode image: %s (%s)\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    rgb.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    return true;
}

// production dispatch for one in-memory image (server uploads, rt stream JPEG
// frames)
bool decode_mem(const std::string & bytes, int & w, int & h, std::vector<uint8_t> & rgb) {
#ifdef VQA_HAVE_LIBJPEG
    if (has_soi((const uint8_t *)bytes.data(), bytes.size())) {
        std::string err;
        if (jpeg_decode_mem((const uint8_t *)bytes.data(), bytes.size(), rgb, w, h, err)) return true;
        fprintf(stderr, "libjpeg decode failed (%s), falling back to stb\n", err.c_str());
    }
#endif
    int comp = 0;
    unsigned char * data = stbi_load_from_memory((const stbi_uc *)bytes.data(), (int)bytes.size(), &w, &h, &comp, 3);
    if (!data) return false;
    rgb.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    return true;
}

}  // namespace

bool load_image(const std::string & path, MediaInput & out) {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (!decode_file(path, w, h, rgb)) return false;
    set_single_frame(out, w, h, std::move(rgb));
    return true;
}

bool load_image_from_memory(const std::string & bytes, MediaInput & out) {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (!decode_mem(bytes, w, h, rgb)) return false;
    set_single_frame(out, w, h, std::move(rgb));
    return true;
}

bool load_frame_dir(const std::string & dir, double fps, MediaInput & out) {
    std::vector<std::string> files;
#ifdef _WIN32
    std::string pattern = dir + "/*";
    WIN32_FIND_DATAA fd;
    HANDLE hfind = FindFirstFileA(pattern.c_str(), &fd);
    if (hfind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "cannot list frame directory: %s\n", dir.c_str());
        return false;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string name = fd.cFileName;
        std::string low = name;
        for (auto & c : low) c = (char)tolower((unsigned char)c);
        if (low.size() < 4) continue;
        if (low.rfind(".png") == low.size() - 4 || low.rfind(".jpg") == low.size() - 4 ||
            low.rfind(".bmp") == low.size() - 4) {
            files.push_back(name);
        }
    } while (FindNextFileA(hfind, &fd));
    FindClose(hfind);
#else
    (void)fps;
    fprintf(stderr, "frame directories are only supported on Windows for now\n");
    return false;
#endif
    if (files.empty()) {
        fprintf(stderr, "no frames in %s\n", dir.c_str());
        return false;
    }
    std::sort(files.begin(), files.end());

    out.kind = MediaKind::Video;
    out.fps = fps > 0 ? fps : 1.0;
    out.frames.clear();
    int W = 0, H = 0;
    for (size_t i = 0; i < files.size(); i++) {
        int w = 0, h = 0;
        std::vector<uint8_t> rgb;
        if (!decode_file(dir + "/" + files[i], w, h, rgb)) {
            fprintf(stderr, "cannot decode %s\n", files[i].c_str());
            return false;
        }
        if (i == 0) { W = w; H = h; }
        if (w != W || h != H) {
            fprintf(stderr, "frame size mismatch in %s\n", dir.c_str());
            return false;
        }
        Frame f;
        f.rgb = std::move(rgb);
        out.frames.push_back(std::move(f));
    }
    out.width = W;
    out.height = H;
    out.timestamps.resize(out.frames.size());
    for (size_t i = 0; i < out.frames.size(); i++) out.timestamps[i] = (double)i / out.fps;
    return true;
}

}  // namespace vqa

#ifdef MEDIA_IO_DECODE_DUMP_MAIN
// Dev tool for decode-engine A/B tests (scripts/test_jpeg_decode.py): dumps
// the raw RGB pixels the production decode path produces for one image.
//   media-decode-dump <image.(png|jpg|bmp)> <out.rgb> [engine|stb]
// engine (default) = the production dispatch (libjpeg for JPEG when built in);
// "stb" forces the stb_image path even for JPEG (the pre-libjpeg reference).
// Output format: int32 LE width, int32 LE height, then width*height*3 RGB8.
int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <image> <out.rgb> [engine|stb]\n", argv[0]);
        return 2;
    }
    const bool force_stb = argc > 3 && strcmp(argv[3], "stb") == 0;
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (force_stb) {
        int comp = 0;
        unsigned char * data = stbi_load(argv[1], &w, &h, &comp, 3);
        if (!data) {
            fprintf(stderr, "stb cannot decode %s (%s)\n", argv[1], stbi_failure_reason());
            return 1;
        }
        rgb.assign(data, data + (size_t)w * h * 3);
        stbi_image_free(data);
    } else {
        vqa::MediaInput in;
        if (!vqa::load_image(argv[1], in) || in.frames.size() != 1) return 1;
        w = in.width;
        h = in.height;
        rgb = std::move(in.frames[0].rgb);
    }
    FILE * f = fopen(argv[2], "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    int32_t dims[2] = {w, h};
    fwrite(dims, sizeof(int32_t), 2, f);
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
    fprintf(stderr, "dumped %dx%d rgb via %s -> %s\n", w, h, force_stb ? "stb" : "engine", argv[2]);
    return 0;
}
#endif
