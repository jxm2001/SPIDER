add_library(spider_tf32_host STATIC tf32_host.cpp)
target_include_directories(spider_tf32_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(spider_tf32_gpu STATIC gpu_2d_7r_tf32.cu)
target_link_libraries(spider_tf32_gpu PUBLIC spider_tf32_host)
target_compile_options(spider_tf32_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

add_executable(stencil_2d_tf32_sparse 2d_stencil_tf32.cpp)
target_link_libraries(stencil_2d_tf32_sparse PRIVATE spider_tf32_gpu)

if(BUILD_TESTING)
    add_executable(test_tf32_host tests/test_tf32_host.cpp)
    target_link_libraries(test_tf32_host PRIVATE spider_tf32_host)
    add_test(NAME tf32_host COMMAND test_tf32_host)

    add_executable(test_tf32_gpu tests/test_tf32_gpu.cu)
    target_link_libraries(test_tf32_gpu PRIVATE spider_tf32_gpu)
    add_test(NAME tf32_gpu COMMAND test_tf32_gpu)
    set_tests_properties(tf32_gpu PROPERTIES SKIP_RETURN_CODE 77)

    foreach(invalid_case IN ITEMS rows columns times)
        if(invalid_case STREQUAL "rows")
            set(invalid_args -M 65 -N 128 -T 1)
        elseif(invalid_case STREQUAL "columns")
            set(invalid_args -M 64 -N 129 -T 1)
        else()
            set(invalid_args -M 64 -N 128 -T 0)
        endif()
        add_test(NAME tf32_invalid_${invalid_case}
                 COMMAND stencil_2d_tf32_sparse ${invalid_args} --check)
        set_tests_properties(tf32_invalid_${invalid_case} PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()
