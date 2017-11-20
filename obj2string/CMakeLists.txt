file(GLOB SOURCES "*.cu" "*.cpp")
cuda_add_library(obj2string SHARED ${SOURCES})

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

add_custom_command(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} build_ext --user -L. -lobj2string
)

add_custom_command(
    TARGET obj2string
    POST_BUILD
    COMMAND python ${SETUP_PY} install --user
)

if(NOT EXISTS  "${CMAKE_INSTALL_PREFIX}/lib")
    message( STATUS "Before installantion, execute:  mkdir ${CMAKE_INSTALL_PREFIX}/lib")    
endif(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/lib") 
    
install(
    TARGETS obj2string 
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
)

if(DEFINED ENV{LD_LYBRARY_PATH})
    if(NOT ENV{LD_LYBRARY_PATH} MATCHES "^(${CMAKE_INSTALL_PREFIX}/lib)+")
        message( STATUS "After installation, you need to execute: export LD_LYBRARY_PATH='${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}'")
        set(ENV{LD_LYBRARY_PATH} "${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}")
    endif()
else()
    message( STATUS "LD_LYBRARY_PATH not defined! After installation, run: export LD_LYBRARY_PATH='${CMAKE_INSTALL_PREFIX}/lib'")   
    set(ENV{LD_LYBRARY_PATH} "${CMAKE_INSTALL_PREFIX}/lib:${LD_LYBRARY_PATH}")
endif()
