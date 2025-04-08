# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

function(add_gtest_target TEST_NAME)
  set(TARGET "test-${TEST_NAME}")
  add_executable(${TARGET})
  target_sources(${TARGET}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/${TEST_NAME}.cc)
  target_include_directories(${TARGET}
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${GTEST_INCLUDE_DIR})
  target_link_libraries(${TARGET}
    PRIVATE
      ${PROJECT_NAME} GTest::gtest GTest::gmock GTest::gtest_main)
  gtest_discover_tests(${TARGET})
endfunction()

add_gtest_target(base)
add_gtest_target(filter)
add_gtest_target(operator)
