#pragma once
// Small ASCII text helpers shared across modules.

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace rope::core {

// Lowercases ASCII letters only; non-ASCII bytes pass through unchanged.
inline std::string to_lower_ascii(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Case-insensitive (ASCII) equality.
inline bool iequals_ascii(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](unsigned char x, unsigned char y) {
               return std::tolower(x) == std::tolower(y);
           });
}

} // namespace rope::core
