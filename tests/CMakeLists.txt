file(GLOB SOURCES "*.cu")
cuda_add_executable(tests ${SOURCES})
#target_include_directories(tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/obj2string)
target_link_libraries(tests ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES} obj2string)

