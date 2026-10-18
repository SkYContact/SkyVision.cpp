// video_pipe.cpp - built-in video decoding via an ffmpeg pipe.
//
// The engine still carries no codec; instead of requiring a pre-extracted frame
// directory, --video streams raw RGB24 frames straight out of ffmpeg. Sampling
// follows the qwen_vl_utils / mlx-vlm convention: n = min(duration * fps,
// max_frames); when the cap binds, the effective fps becomes max_frames /
// duration so the sampled frames stay spread uniformly over the whole video.
#include "media_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>

#ifdef _WIN32
  #define popen _popen
  #define pclose _pclose
#endif

namespace vqa {

static std::string find_ffmpeg() {
    if (const char * e = getenv("FFMPEG")) {
        std::string p = e;
#ifdef _WIN32
        // cmd.exe cannot launch a quoted command whose path uses forward slashes
        std::replace(p.begin(), p.end(), '/', '\\');
#endif
        return p;
    }
#ifdef _WIN32
    const char * candidates[] = { "ffmpeg.exe", "ffmpeg" };
#else
    const char * candidates[] = { "ffmpeg" };
#endif
    for (const char * c : candidates) {
#ifdef _WIN32
        std::string cmd = std::string(c) + " -version 2>NUL >NUL";
#else
        std::string cmd = std::string(c) + " -version 2>/dev/null >/dev/null";
#endif
        if (system(cmd.c_str()) == 0) return c;
    }
    return {};
}

// one probe pass: duration + first video stream size from ffmpeg's stderr dump
// (avoids a ffprobe dependency)
static bool probe_metadata(const std::string & ffmpeg, const std::string & path,
                           double & duration, int & w, int & h) {
#ifdef _WIN32
    // wrap the whole line in an extra quote pair: cmd /c otherwise strips the leading
    // quote of a quoted exe path and mangles the command
    std::string cmd = "\"\"" + ffmpeg + "\" -i \"" + path + "\" -f null - 2>&1\"";
#else
    std::string cmd = "\"" + ffmpeg + "\" -i \"" + path + "\" -f null - 2>&1";
#endif
    FILE * p = popen(cmd.c_str(), _WIN32 ? "rb" : "r");
    if (!p) return false;
    std::string buf;
    char tmp[512];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), p)) > 0) buf.append(tmp, n);
    pclose(p);

    size_t at = buf.find("Duration:");
    if (at != std::string::npos) {
        int hh = 0, mm = 0;
        double ss = 0.0;
        if (sscanf(buf.c_str() + at + 9, "%d:%d:%lf", &hh, &mm, &ss) == 3)
            duration = hh * 3600 + mm * 60 + ss;
    }
    size_t vs = buf.find("Video:");
    if (vs != std::string::npos) {
        size_t p1 = buf.find_first_of("0123456789", vs);
        while (p1 != std::string::npos) {
            int a = 0, c = 0;
            char x = 0;
            if (sscanf(buf.c_str() + p1, "%d%c%d", &a, &x, &c) == 3 && x == 'x' && a > 0 && c > 0) {
                w = a; h = c;
                break;
            }
            p1 = buf.find_first_of("0123456789", p1 + 1);
        }
    }
    return duration > 0 && w > 0 && h > 0;
}

bool load_video_file(const std::string & path, double fps, int max_frames, MediaInput & out) {
    const std::string ffmpeg = find_ffmpeg();
    if (ffmpeg.empty()) {
        fprintf(stderr, "ffmpeg not found (install it or set FFMPEG=<path>); "
                        "or pre-extract frames and use --frames <dir>\n");
        return false;
    }
    double duration = 0;
    int w = 0, h = 0;
    if (!probe_metadata(ffmpeg, path, duration, w, h)) {
        fprintf(stderr, "cannot probe video metadata: %s\n", path.c_str());
        return false;
    }

    if (fps <= 0) fps = 1.0;
    if (max_frames <= 0) max_frames = 32;
    // cap binds -> resample uniformly across the whole video (qwen_vl_utils style)
    double eff_fps = fps;
    if (duration * fps > max_frames) eff_fps = max_frames / duration;
    const int n = std::max(1, (int)(duration * eff_fps + 0.5));

    char vf[128];
    snprintf(vf, sizeof(vf), "fps=%.4f", eff_fps);
#ifdef _WIN32
    std::string cmd = "\"\"" + ffmpeg + "\" -nostdin -hide_banner -loglevel error -i \"" + path + "\" -vf " + vf +
                      " -frames:v " + std::to_string(n) +
                      " -pix_fmt rgb24 -f rawvideo -\"";
#else
    std::string cmd = "\"" + ffmpeg + "\" -nostdin -hide_banner -loglevel error -i \"" + path + "\" -vf " + vf +
                      " -frames:v " + std::to_string(n) +
                      " -pix_fmt rgb24 -f rawvideo -";
#endif
    FILE * p = popen(cmd.c_str(), _WIN32 ? "rb" : "r");   // glibc popen rejects "rb" (EINVAL)
    if (!p) { fprintf(stderr, "cannot spawn ffmpeg (errno %d): %s\n", errno, cmd.c_str()); return false; }

    out.kind = MediaKind::Video;
    out.width = w;
    out.height = h;
    out.fps = eff_fps;
    out.frames.clear();
    out.timestamps.clear();

    const size_t need = (size_t)w * h * 3;
    std::vector<unsigned char> pending;
    char tmp[1 << 16];
    size_t got;
    while ((got = fread(tmp, 1, sizeof(tmp), p)) > 0) {
        pending.insert(pending.end(), tmp, tmp + got);
        while (pending.size() >= need) {
            Frame f;
            f.rgb.assign(pending.begin(), pending.begin() + need);
            pending.erase(pending.begin(), pending.begin() + need);
            out.timestamps.push_back(out.frames.size() / eff_fps);
            out.frames.push_back(std::move(f));
        }
    }
    pclose(p);
    if (out.frames.empty()) { fprintf(stderr, "no frames decoded from %s\n", path.c_str()); return false; }
    fprintf(stderr, "video: %.1fs @ %.2f fps -> %d frames %dx%d\n",
            duration, eff_fps, (int)out.frames.size(), out.width, out.height);
    return true;
}


// Decode any ffmpeg-readable audio file to mono 16 kHz float PCM. The models
// that take audio (Qwen2.5-Omni) expect exactly this rate, so it is baked in
// here rather than spread over call sites.
bool load_audio(const std::string & path, MediaInput & out) {
    const std::string ffmpeg = find_ffmpeg();
    if (ffmpeg.empty()) {
        fprintf(stderr, "ffmpeg not found (install it or set FFMPEG=<path>)\n");
        return false;
    }
#ifdef _WIN32
    std::string cmd = "\"\"" + ffmpeg + "\" -nostdin -hide_banner -loglevel error -i \"" + path +
                      "\" -ac 1 -ar 16000 -f f32le -\"";
#else
    std::string cmd = "\"" + ffmpeg + "\" -nostdin -hide_banner -loglevel error -i \"" + path +
                      "\" -ac 1 -ar 16000 -f f32le -";
#endif
    FILE * p = popen(cmd.c_str(), _WIN32 ? "rb" : "r");
    if (!p) { fprintf(stderr, "cannot spawn ffmpeg (errno %d)\n", errno); return false; }
    std::vector<float> pcm;
    float buf[8192];
    size_t n;
    while ((n = fread(buf, sizeof(float), 8192, p)) > 0)
        pcm.insert(pcm.end(), buf, buf + n);
    const int rc = pclose(p);
    if (rc != 0 || pcm.empty()) {
        fprintf(stderr, "audio decode failed (%s)\n", path.c_str());
        return false;
    }
    out.audio = std::move(pcm);
    out.sample_rate = 16000;
    fprintf(stderr, "audio: %s -> %.1f s @ 16 kHz\n", path.c_str(), out.audio.size() / 16000.0);
    return true;
}

}  // namespace vqa
