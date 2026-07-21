#pragma once
// INI-style config file reader ([section] / key = value, # comments); keys accessed as "section.key".

#include <filesystem>
#include <map>
#include <string>

namespace rope::io {

class ConfigReader {
public:
    explicit ConfigReader(const std::filesystem::path& path);

    bool has(std::string_view key) const noexcept;

    // Required key — throws if absent.
    std::string get(std::string_view key) const;

    // Optional key — returns def if absent.
    std::string get(std::string_view key, std::string_view def) const;
    int         get_int(std::string_view key, int def = 0) const;
    double      get_double(std::string_view key, double def = 0.0) const;

private:
    std::map<std::string, std::string> data_;

    static std::string trim(const std::string& s);
};

} // namespace rope::io
