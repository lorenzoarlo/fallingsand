cmake_minimum_required(VERSION 3.10)
project(FallingSandProject)



# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable CUDA logic
option(ENABLE_CUDA "Enable CUDA simulation logic" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

file(GLOB CORE_SOURCES "src/*.c")
# Remove logic.c 
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/logic.c")

if(ENABLE_CUDA)
    message(STATUS "CUDA ENABLED")
    enable_language(CUDA)
    if(NOT DEFINED CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "75")
    else()
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    endif()
    if(NOT DEFINED SIMULATION_LOGIC OR SIMULATION_LOGIC MATCHES "^.+\.c$")
        set(SIMULATION_LOGIC "src/cuda/super-optimized")
    endif()
    message(STATUS "Compiling with LOGIC=${SIMULATION_LOGIC}")
    file(GLOB CUDA_SOURCES "src/cuda/super-optimized/*.cu")
    add_executable(fallingsand
        ${CORE_SOURCES}
        ${CUDA_SOURCES}
    )
    set_target_properties(fallingsand PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_features(fallingsand PRIVATE cuda_std_11)
else()
    if(NOT DEFINED SIMULATION_LOGIC)
        set(SIMULATION_LOGIC "src/logic.c")
    endif()
    
    message(STATUS "Using simulation logic from: ${SIMULATION_LOGIC}")
    
    add_executable(fallingsand
        ${CORE_SOURCES}
        ${SIMULATION_LOGIC}
    )

    # Check if SIMULATION_LOGIC is a .cpp file to set C++ compilation
    if (SIMULATION_LOGIC MATCHES "\\.cpp$")
        message(STATUS "Detected C++ SIMD logic. Linking Highway...")
        # Search for Highway library
        find_library(HWY_LIB hwy)
        find_path(HWY_INCLUDE_DIR hwy/highway.h)

        if(HWY_LIB AND HWY_INCLUDE_DIR)
            message(STATUS "Found Google Highway library: ${HWY_LIB}")
            message(STATUS "Found Google Highway headers: ${HWY_INCLUDE_DIR}")

            target_link_libraries(fallingsand PRIVATE ${HWY_LIB})
            target_include_directories(fallingsand PRIVATE ${HWY_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "Google Highway library (libhwy) not found! Install it via 'brew install highway' or apt.")
        endif()
    endif()
endif()
# only link math library on Unix-like systems
# target_link_libraries(fallingsand m)
