#include "decoder_factory.h"
#include "coae_decoder.h"

#include <stdexcept>

namespace rope::forecast {

std::unique_ptr<IDecoder> make_decoder(
    const std::filesystem::path& dir,
    const io::ModelManifest&     manifest,
    const Config&                cfg)
{
    if (manifest.decoder_kind == "coae")
        return std::make_unique<CoaeDecoder>(dir, manifest, cfg);

    throw std::runtime_error(
        "make_decoder: unrecognized decoder.kind '" + manifest.decoder_kind + "'");
}

} // namespace rope::forecast
