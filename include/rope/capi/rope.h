/* ROPE C API — memory-maps the cache written by `rope forecast`; query via rope_query/rope_query_batch. */

#ifndef ROPE_CAPI_H
#define ROPE_CAPI_H

#include "rope/core/export.h"

#define ROPE_CAPI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle — heap-allocated by rope_open, freed by rope_close. */
typedef struct rope_interp rope_interp_t;

/* Opens the cache file (cache_path=NULL: platform default). Returns a handle, or NULL with err_buf filled on failure. */
ROPE_API rope_interp_t* rope_open(const char* cache_path,
                                   char*       err_buf,
                                   int         err_len);

/* Queries density/uncertainty (kg/m³) at one point: time_unix (UTC seconds), lst [0,24), lat/alt_km within grid range. Returns ROPE_OK or an error code. */
ROPE_API int rope_query(rope_interp_t* interp,
                         int            mode,
                         double         time_unix,
                         double         lst,
                         double         lat,
                         double         alt_km,
                         double*        density,
                         double*        uncertainty,
                         char*          err_buf,
                         int            err_len);

/* Batched rope_query over n points (caller-allocated output arrays). Stops and returns non-zero at the first failing point. */
ROPE_API int rope_query_batch(rope_interp_t* interp,
                               int            mode,
                               int            n,
                               const double*  times_unix,
                               const double*  lst,
                               const double*  lat,
                               const double*  alt_km,
                               double*        density,
                               double*        uncertainty,
                               char*          err_buf,
                               int            err_len);

/* Releases the handle (unmaps the cache file). Safe to call with NULL. */
ROPE_API void rope_close(rope_interp_t* interp);

/* Mode constants */
#define ROPE_HOLD   0
#define ROPE_INTERP 1

/* Return codes */
#define ROPE_OK                  0
#define ROPE_ERR_NO_FORECAST     2   /* no forecast cached at this path -- run `rope forecast` first */
#define ROPE_ERR_TIME_RANGE      3   /* query time outside forecast window */
#define ROPE_ERR_SPATIAL_RANGE   4   /* query point outside grid bounds */
#define ROPE_ERR_BAD_ARG         5   /* invalid argument (NULL pointer, etc.) */
#define ROPE_ERR_INTERNAL        6   /* unexpected internal failure */
#define ROPE_ERR_CACHE_CORRUPT   7   /* cache file exists but is corrupt or from an incompatible build */

#ifdef __cplusplus
}
#endif

#endif /* ROPE_CAPI_H */
