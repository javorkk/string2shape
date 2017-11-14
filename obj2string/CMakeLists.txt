file(GLOB SOURCES "*.cu" "*.cpp")
cuda_add_library(obj2string SHARED ${SOURCES})

ADD_CUSTOM_COMMAND(
    TARGET obj2string
    POST_BUILD
    COMMAND python setup.py build_ext -lobj2string
)