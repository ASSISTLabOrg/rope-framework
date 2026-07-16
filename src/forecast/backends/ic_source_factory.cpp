#include "ic_source_factory.h"
#include "lookup_table_ic_source.h"

#include "rope/io/ic_table.h"

#include <stdexcept>

namespace rope::forecast {

std::unique_ptr<IICSource> make_ic_source(
    const std::filesystem::path& dir,
    const std::string&           ic_kind)
{
    if (ic_kind == "ic_lookup_table")
        return std::make_unique<LookupTableICSource>(
            io::ICTable::load_from_dir(dir));

    throw std::runtime_error(
        "make_ic_source: unrecognized ic.kind '" + ic_kind + "'");
}

} // namespace rope::forecast
