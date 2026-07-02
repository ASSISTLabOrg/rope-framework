#pragma once
#include "rope/core/platform.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace rope::wire {

inline void send_msg(platform::IpcSocket& sock, const nlohmann::json& j) {
    std::string s = j.dump();
    auto len = static_cast<std::uint32_t>(s.size());
    std::uint8_t hdr[4] = {
        static_cast<std::uint8_t>(len),
        static_cast<std::uint8_t>(len >>  8),
        static_cast<std::uint8_t>(len >> 16),
        static_cast<std::uint8_t>(len >> 24),
    };
    sock.send_all(hdr, 4);
    sock.send_all(s.data(), s.size());
}

inline nlohmann::json recv_msg(platform::IpcSocket& sock) {
    std::uint8_t hdr[4];
    sock.recv_all(hdr, 4);
    std::uint32_t len = static_cast<std::uint32_t>(hdr[0])
                      | static_cast<std::uint32_t>(hdr[1]) <<  8
                      | static_cast<std::uint32_t>(hdr[2]) << 16
                      | static_cast<std::uint32_t>(hdr[3]) << 24;
    std::string buf(len, '\0');
    sock.recv_all(buf.data(), len);
    return nlohmann::json::parse(buf);
}

} // namespace rope::wire
