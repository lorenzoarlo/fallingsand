cmake_minimum_required(VERSION 3.10)
project(FallingSandProject C)

# Imposta lo standard C
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Opzione per abilitare la logica CUDA
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
    file(GLOB CUDA_SOURCES "src/cuda/naive/*.cu")
    list(REMOVE_ITEM CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/naive/device_functions.cu")
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
endif()
# only link math library on Unix-like systems
# target_link_libraries(fallingsand m)