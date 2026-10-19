// jpeg_decode.cpp - the only translation unit that touches libjpeg's C API.
// See jpeg_decode.hpp for the why (bit-exact parity with PIL / libjpeg family)
// and jpeglib's usual setjmp/longjmp error sidecar for the how.
#include "core/media/jpeg_decode.hpp"

#ifdef VQA_HAVE_LIBJPEG

#include <csetjmp>
#include <cstdio>

#include "jerror.h"
#include "jpeglib.h"

namespace vqa {

namespace {

struct JpegErrMgr {
    jpeg_error_mgr pub;          // libjpeg's error bookkeeping
    jmp_buf jmp;                 // where error_exit lands
    char msg[JMSG_LENGTH_MAX];   // last error, formatted
};

void error_exit(j_common_ptr cinfo) {
    JpegErrMgr * m = reinterpret_cast<JpegErrMgr *>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, m->msg);
    longjmp(m->jmp, 1);
}

// CMYK JPEGs (including YCCK, which libjpeg converts to CMYK): libjpeg has no
// CMYK->RGB conversion, so decode() inverts the four channels itself. NOTE:
// that is a plain 255-x; PIL applies the Adobe inverted-CMYK convention, so
// CMYK pixels can differ from PIL (recorded known difference, see the roadmap).
bool decode(const uint8_t * data, size_t size,
            std::vector<uint8_t> & rgb, int & width, int & height,
            std::string & err) {
    jpeg_decompress_struct cinfo{};
    JpegErrMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;
    if (setjmp(jerr.jmp)) {
        err = jerr.msg;
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<unsigned char *>(data), (unsigned long)size);
    jpeg_read_header(&cinfo, TRUE);

    const bool cmyk = cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK;
    if (!cmyk) cinfo.out_color_space = JCS_RGB;  // libjpeg's islow IDCT = PIL's

    jpeg_start_decompress(&cinfo);
    width = (int)cinfo.output_width;
    height = (int)cinfo.output_height;
    const size_t comps = cinfo.output_components;  // 3 for RGB, 4 for CMYK
    rgb.resize((size_t)width * height * 3);
    const size_t row_px = (size_t)width;
    std::vector<JSAMPLE> cmyk_row;
    JSAMPLE * raw_row = nullptr;
    if (cmyk) {
        cmyk_row.resize(row_px * comps);
        raw_row = cmyk_row.data();
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t * dst = rgb.data() + (size_t)cinfo.output_scanline * row_px * 3;
        if (!cmyk) {
            JSAMPROW row = dst;
            jpeg_read_scanlines(&cinfo, &row, 1);
        } else {
            JSAMPROW row = raw_row;
            jpeg_read_scanlines(&cinfo, &row, 1);
            for (size_t x = 0; x < row_px; x++) {
                const JSAMPLE * px = raw_row + x * comps;
                dst[x * 3 + 0] = (uint8_t)(255 - px[0]);
                dst[x * 3 + 1] = (uint8_t)(255 - px[1]);
                dst[x * 3 + 2] = (uint8_t)(255 - px[2]);
            }
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}

}  // namespace

bool jpeg_decode_mem(const uint8_t * data, size_t size,
                     std::vector<uint8_t> & rgb, int & width, int & height,
                     std::string & err) {
    if (!looks_like_jpeg(data, size)) {
        err = "not a JPEG stream (missing SOI marker)";
        return false;
    }
    rgb.clear();
    width = height = 0;
    return decode(data, size, rgb, width, height, err);
}

bool jpeg_decode_file(const std::string & path,
                      std::vector<uint8_t> & rgb, int & width, int & height,
                      std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    bool ok = ferror(f) == 0;
    fclose(f);
    if (!ok) {
        err = "read error on " + path;
        return false;
    }
    return jpeg_decode_mem(bytes.data(), bytes.size(), rgb, width, height, err);
}

}  // namespace vqa

#endif  // VQA_HAVE_LIBJPEG
