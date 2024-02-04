# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include)

add_subdirectory(common)
add_subdirectory(ops)
add_subdirectory(c_api)

# add_library(arrayfire OBJECT)
# target_sources(arrayfire PRIVATE $<TARGET_OBJECTS:arrayfire_c_api_src> $<TARGET_OBJECTS:arrayfire_common_src> $<TARGET_OBJECTS:arrayfire_ops_src>)

# add_library(arrayfire STATIC)
# target_link_libraries(arrayfire PUBLIC -lcudart c_api_src common_src ops_src)
# install(
# TARGETS
# arrayfire
# LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
# )
