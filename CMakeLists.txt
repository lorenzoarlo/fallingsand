cmake_minimum_required(VERSION 3.10)
project(FallingSandSimulation)

set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Define source files for the core engine
set(CORE_SOURCES
    src/main.c
    src/universe.c
    src/universe.h
    src/io.c
    src/io.h
    src/graphics.c
    src/graphics.h
    src/cli.c
    src/cli.h
    src/test.c
    src/test.h
    src/simulation.h
)

# Variable to select the logic implementation file.
# Default is "src/logic_basic.c" but can be overridden via command line:
# cmake -DSIMULATION_LOGIC="src/logic_advanced.c" ..
if(NOT DEFINED SIMULATION_LOGIC)
    set(SIMULATION_LOGIC "src/logic.c")
endif()

message(STATUS "Using simulation logic from: ${SIMULATION_LOGIC}")

# Create the executable
add_executable(fallingsand
    ${CORE_SOURCES}
    ${SIMULATION_LOGIC}
)

# Create output directory if it doesn't exist (optional helper)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)