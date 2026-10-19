// aligned_buf.hpp - a byte buffer whose data() is 64-byte aligned.
//
// ggml_backend_cpu_buffer_from_ptr requires TENSOR_ALIGNMENT (32) and
// std::vector's data() guarantees nothing beyond alignof(char) - glibc's
// mmap-backed large blocks land at page+16. Every backend that stages a tensor
// on the host uses this instead of std::vector<char>.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vqa {

struct AlignedBuf {
    void resize(size_t n) {
        raw.resize(n + kAlign);
        ptr = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(raw.data()) + (kAlign - 1)) & ~uintptr_t(kAlign - 1));
        len = n;
    }
    bool empty() const { return len == 0; }
    size_t size() const { return len; }
    char * data() { return ptr; }
    const char * data() const { return ptr; }
    void clear() { raw.clear(); ptr = nullptr; len = 0; }
    void shrink_to_fit() { raw.shrink_to_fit(); }

private:
    static constexpr size_t kAlign = 64;
    std::vector<char> raw;
    char * ptr = nullptr;
    size_t len = 0;
};

}  // namespace vqa
