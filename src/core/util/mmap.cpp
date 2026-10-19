#include "core/util/mmap.hpp"

#include <cstdio>

#include <sys/stat.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
#endif

namespace vqa {

bool MMap::load(const char * path) {
#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER li;
    if (!GetFileSizeEx(hf, &li) || li.QuadPart <= 0) { CloseHandle(hf); return false; }
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) { CloseHandle(hf); return false; }
    void * p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hm); CloseHandle(hf); return false; }
    ptr = p;
    size = (size_t)li.QuadPart;
    hfile = hf;
    hmap = hm;
    return true;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    ptr = p;
    size = (size_t)st.st_size;
    return true;
#endif
}

bool MMap::map_file(const char * path) { return load(path); }

void MMap::release() {
#ifdef _WIN32
    if (ptr) { UnmapViewOfFile(ptr); ptr = nullptr; }
    if (hmap) { CloseHandle((HANDLE)hmap); hmap = nullptr; }
    if (hfile) { CloseHandle((HANDLE)hfile); hfile = nullptr; }
#else
    if (ptr) { munmap(ptr, size); ptr = nullptr; }
#endif
    size = 0;
}

bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    out.resize((size_t)sz);
    size_t got = fread(out.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { out.clear(); return false; }
    return true;
}

}  // namespace vqa
