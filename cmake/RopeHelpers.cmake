# rope_set_runtime_rpath(target)
#
# Sets BUILD_RPATH on <target> to the ORT (and, if found, LibTorch) lib dirs
# so the binary can be run in-place by CTest without setting LD_LIBRARY_PATH.
# On Windows, copies the runtime DLLs next to the target binary instead.
#
# Requires ORT_LIB to be set. No-op if it isn't.
function(rope_set_runtime_rpath target)
    if(NOT ORT_LIB)
        return()
    endif()

    get_filename_component(_rpath "${ORT_LIB}" DIRECTORY)

    if(Torch_FOUND)
        if(TORCH_INSTALL_PREFIX)
            set(_torch_dir "${TORCH_INSTALL_PREFIX}/lib")
        else()
            list(GET TORCH_LIBRARIES 0 _first_torch)
            get_filename_component(_torch_dir "${_first_torch}" DIRECTORY)
        endif()
        list(APPEND _rpath "${_torch_dir}")
    endif()

    set_target_properties(${target} PROPERTIES BUILD_RPATH "${_rpath}")

    if(WIN32)
        file(GLOB _ort_dlls "${CMAKE_INSTALL_BINDIR}/${_rpath}/*.dll")
        # Derive dir from ORT_LIB for the glob (BUILD_RPATH only helps Unix).
        get_filename_component(_ort_dir "${ORT_LIB}" DIRECTORY)
        file(GLOB _ort_dlls "${_ort_dir}/*.dll")
        foreach(_dll ${_ort_dlls})
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM
            )
        endforeach()
        if(Torch_FOUND AND DEFINED _torch_dir)
            file(GLOB _torch_dlls "${_torch_dir}/*.dll")
            foreach(_dll ${_torch_dlls})
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_dll}" "$<TARGET_FILE_DIR:${target}>"
                    VERBATIM
                )
            endforeach()
        endif()
    endif()
endfunction()
