#include "media_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <vector>

// the single translation unit that owns the stb_image implementation
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace vqa {

bool load_image(const std::string & path, MediaInput & out) {
    int w = 0, h = 0, comp = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!data) {
        fprintf(stderr, "cannot decode image: %s (%s)\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    out.kind = MediaKind::Image;
    out.width = w;
    out.height = h;
    out.fps = 0.0;
    out.timestamps.clear();
    out.frames.clear();
    Frame f;
    f.rgb.assign(data, data + (size_t)w * h * 3);
    out.frames.push_back(std::move(f));
    stbi_image_free(data);
    return true;
}

bool load_image_from_memory(const std::string & bytes, MediaInput & out) {
    int w = 0, h = 0, comp = 0;
    unsigned char * data = stbi_load_from_memory((const stbi_uc *)bytes.data(), (int)bytes.size(), &w, &h, &comp, 3);
    if (!data) return false;
    out.kind = MediaKind::Image;
    out.width = w;
    out.height = h;
    out.fps = 0.0;
    out.timestamps.clear();
    out.frames.clear();
    Frame f;
    f.rgb.assign(data, data + (size_t)w * h * 3);
    out.frames.push_back(std::move(f));
    stbi_image_free(data);
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
        int w = 0, h = 0, comp = 0;
        unsigned char * data = stbi_load((dir + "/" + files[i]).c_str(), &w, &h, &comp, 3);
        if (!data) {
            fprintf(stderr, "cannot decode %s\n", files[i].c_str());
            return false;
        }
        if (i == 0) { W = w; H = h; }
        if (w != W || h != H) {
            fprintf(stderr, "frame size mismatch in %s\n", files[i].c_str());
            stbi_image_free(data);
            return false;
        }
        Frame f;
        f.rgb.assign(data, data + (size_t)w * h * 3);
        out.frames.push_back(std::move(f));
        stbi_image_free(data);
    }
    out.width = W;
    out.height = H;
    out.timestamps.resize(out.frames.size());
    for (size_t i = 0; i < out.frames.size(); i++) out.timestamps[i] = (double)i / out.fps;
    return true;
}

}  // namespace vqa
