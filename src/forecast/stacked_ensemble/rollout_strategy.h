#pragma once
#include "backends/model_interface.h"

namespace rope::forecast {

class IRolloutStrategy {
public:
    virtual ~IRolloutStrategy() = default;

    // Run auto-regressive rollout for a single base model.
    //   model     — base temporal model
    //   x_chunk   — (H+1, S, D) flat float array (read-only)
    //   H         — forecast horizon
    //   preds_out — (H, K) flat float array (caller allocates)
    virtual void run(
        IModel&      model,
        const float* x_chunk,
        int          H,
        float*       preds_out
    ) const = 0;
};

} // namespace rope::forecast
