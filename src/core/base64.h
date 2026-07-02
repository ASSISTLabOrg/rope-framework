#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rope::detail {

inline std::string base64_encode(const void* data, std::size_t n_bytes) {
    static constexpr char kAlpha[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::string out;
    out.reserve((n_bytes + 2) / 3 * 4);
    for (std::size_t i = 0; i < n_bytes; i += 3) {
        std::uint32_t b = static_cast<std::uint32_t>(p[i]) << 16;
        if (i + 1 < n_bytes) b |= static_cast<std::uint32_t>(p[i + 1]) << 8;
        if (i + 2 < n_bytes) b |= static_cast<std::uint32_t>(p[i + 2]);
        out.push_back(kAlpha[(b >> 18) & 0x3F]);
        out.push_back(kAlpha[(b >> 12) & 0x3F]);
        out.push_back((i + 1 < n_bytes) ? kAlpha[(b >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < n_bytes) ? kAlpha[b & 0x3F]        : '=');
    }
    return out;
}

template<typename T>
inline std::string base64_encode(const std::vector<T>& v) {
    return base64_encode(v.data(), v.size() * sizeof(T));
}

inline std::vector<std::uint8_t> base64_decode(std::string_view in) {
    static constexpr unsigned char kTable[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64,255,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
    };

    std::vector<std::uint8_t> out;
    out.reserve(in.size() * 3 / 4);

    unsigned int buf  = 0;
    int          bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (c >= 128) throw std::runtime_error("base64_decode: invalid char");
        unsigned char v = kTable[c];
        if (v == 64) continue;  // whitespace
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace rope::detail
