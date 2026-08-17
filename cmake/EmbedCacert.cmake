# Embeds cmake/cacert.pem into generated/cacert_pem.cpp as chunked raw string literals.
# MSVC caps a single string literal at 65535 bytes (C2026); cacert.pem is ~189 KB, so it's
# split into <64 KB pieces and reassembled via adjacent string-literal concatenation.

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cacert.pem" _rope_cacert_content)
string(LENGTH "${_rope_cacert_content}" _rope_cacert_len)

set(_rope_cacert_chunk_size 16000)
set(_rope_cacert_chunks "")
set(_rope_cacert_pos 0)
while(_rope_cacert_pos LESS _rope_cacert_len)
    string(SUBSTRING "${_rope_cacert_content}" ${_rope_cacert_pos} ${_rope_cacert_chunk_size} _rope_cacert_chunk)
    string(APPEND _rope_cacert_chunks "R\"ROPE_CACERT(${_rope_cacert_chunk})ROPE_CACERT\"\n")
    math(EXPR _rope_cacert_pos "${_rope_cacert_pos} + ${_rope_cacert_chunk_size}")
endwhile()

set(ROPE_CACERT_PEM_CHUNKS "${_rope_cacert_chunks}")
configure_file(cmake/cacert_pem.cpp.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cacert_pem.cpp" @ONLY)
