#pragma once
// Lightweight column-oriented CSV reader. Reads the whole file on construction; columns accessed by name.

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace rope::io {

class CsvReader {
public:
    explicit CsvReader(const std::filesystem::path& path, char delim = ',');

    std::size_t nrows() const noexcept { return n_rows_; }
    bool has_column(std::string_view name) const noexcept;
    const std::vector<std::string>& column_names() const noexcept { return headers_; }

    const std::string& get(std::string_view col, std::size_t row) const;
    float              get_float(std::string_view col, std::size_t row) const;
    int                get_int(std::string_view col, std::size_t row) const;

private:
    std::vector<std::string>                 headers_;
    std::vector<std::vector<std::string>>    columns_;
    std::unordered_map<std::string, std::size_t> index_;
    std::size_t n_rows_ = 0;

    std::size_t col_index(std::string_view name) const;

    static std::string strip(const std::string& s);
    static std::vector<std::string> split(const std::string& s, char delim);
};

} // namespace rope::io
