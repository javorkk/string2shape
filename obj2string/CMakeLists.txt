file(GLOB SOURCES "*.cu" "*.cpp")
cuda_add_library(obj2string SHARED ${SOURCES})

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

add_custom_command(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} build_ext --user -L.
)

add_custom_command(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} install --user --force
)

if(NOT WIN32)
    if(NOT EXISTS  "${CMAKE_INSTALL_PREFIX}/lib")
        message( STATUS "${BoldYellow}Install directory dose not exist! Before installantion, execute: mkdir ${CMAKE_INSTALL_PREFIX}/lib$#{ColourReset}") 
    endif(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/lib")

    install(
        TARGETS obj2string 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    )

#if(DEFINED ENV{LD_LYBRARY_PATH})
#    if(NOT ENV{LD_LYBRARY_PATH} MATCHES "^(${CMAKE_INSTALL_PREFIX}/lib)+")
#        message( STATUS "${BoldYellow}After installation, you might need to execute: export LD_LYBRARY_PATH='${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}'${ColourReset}")
#        set(ENV{LD_LYBRARY_PATH} "${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}")
#    endif()
#else()
#    message( STATUS "${BoldYellow}LD_LYBRARY_PATH not defined! After installation, run: export LD_LYBRARY_PATH='${CMAKE_INSTALL_PREFIX}/lib'${ColourReset}")   
#    set(ENV{LD_LYBRARY_PATH} "${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}")
#endif()

endif(NOT WIN32)


