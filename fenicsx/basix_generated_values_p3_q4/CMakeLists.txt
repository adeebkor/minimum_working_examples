cmake_minimum_required(VERSION 3.16)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_NAME test_stiffness_operator)

set(CMAKE_CXX_FLAGS "-O3 -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-O3 -march=native  ${CMAKE_C_FLAGS} -Wall")

project(${PROJECT_NAME})

# Get DOLFINx configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

include_directories(".")

# Executable
add_executable(${PROJECT_NAME} form.c main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx blas lapack)

message(STATUS ${CMAKE_CXX_FLAGS})