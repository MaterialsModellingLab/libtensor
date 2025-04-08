# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(libtensor LANGUAGES CXX DESCRIPTION "C++ Tensor Library")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local/" CACHE PATH "Default install path" FORCE)
endif()

# Define name for different platforms
set(LIBTENSOR_LIB_NAME "${PROJECT_NAME}")

# Option
option(BUILD_TESTING "Build unit tests" OFF)
option(BUILD_BENCHMARK "Build benchmark" OFF)

# Dependencies
if(CMAKE_CXX_COMPILER_ID STREQUAL "FujitsuClang")
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  set_property(
    TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_LINK_LIBRARIES -fopenmp)
else()
  find_package(OpenMP REQUIRED)
endif()

# Include local module
include(${CMAKE_SOURCE_DIR}/cmake/clang_format.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cppcheck.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/git-version.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/install.cmake)

# Get git tag as PROJECT_VERSION
get_git_version()

set(TARGET ${LIBTENSOR_LIB_NAME})
add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX)
target_compile_features(${TARGET} INTERFACE cxx_std_17)

if(BUILD_TESTING)
  include(CTest)
  add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

library_install(${LIBTENSOR_LIB_NAME})

get_target_property(DEFINES ${LIBTENSOR_LIB_NAME} COMPILE_DEFINITIONS)
message(STATUS "<<< Build configuration >>>
    Version:           ${PROJECT_VERSION}
    Operating System:  ${CMAKE_SYSTEM_NAME} ${CMAKE_LINUX_DISTRO} ${CMAKE_DISTRO_VERSION}
    CMake Version:     ${CMAKE_VERSION}
    Install path:      ${CMAKE_INSTALL_PREFIX}
    Generator:         ${CMAKE_GENERATOR} using ${CMAKE_MAKE_PROGRAM}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" BTYPE)
message(STATUS "<<< Compilers and Flags: >>>
    C++ Compiler:      ${CMAKE_CXX_COMPILER}
    Type:              ${CMAKE_CXX_COMPILER_ID}
    Version:           ${CMAKE_CXX_COMPILER_VERSION}
    C++ Flags:         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BTYPE}}
    Defines:           ${DEFINES}")
