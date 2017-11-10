file(GLOB SOURCES "*.cu" "*.cpp")
cuda_add_library(obj2string SHARED ${SOURCES})
