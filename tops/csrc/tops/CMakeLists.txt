# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include)

# add_subdirectory(common)
add_subdirectory(arrayfire)
add_subdirectory(torch)
add_subdirectory(oneflow)
add_subdirectory(tensorrt_llm)
add_subdirectory(c_api)

add_library(tops STATIC)
target_link_libraries(tops PUBLIC -lcudart -lcurand c_api_src arrayfire_common_src arrayfire_c_api_src arrayfire_ops_src oneflow_c_api_src oneflow_distribution_src oneflow_random_src torch_common_src torch_c_api_src tensorrt_llm_c_api_src tensorrt_llm_common_src tensorrt_llm_kernels_src tensorrt_llm_gemm_src)
install(
    TARGETS
    tops
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
