file(GLOB SOURCES "*.cu")
cuda_add_executable(tests ${SOURCES})
target_link_libraries(tests ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES} obj2string)

