file(GLOB SOURCES "*.cu" "*.cpp")
cuda_add_library(obj2string SHARED ${SOURCES})

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

ADD_CUSTOM_COMMAND(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} build_ext -L. -lobj2string
)

ADD_CUSTOM_COMMAND(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} install --user
)
