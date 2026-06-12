cmake_minimum_required(VERSION 3.10)
project(ANNarchy_CPP_Templates)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0")

# GoogleTest einbinden (z.B. über FetchContent)
include(FetchContent)
if(CMAKE_VERSION VERSION_LESS "3.27")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
else()
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)
endif()
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(../ANNarchy/include/)

add_executable(ANNarchy_CPP_Templates
  # SpMV formats
  spmv_formats/test_MatrixReinitialize.cpp
  spmv_formats/test_Matrix6x8_WithAllRows.cpp
  spmv_formats/test_Matrix6x8_WithEmptyRows.cpp
  spmv_formats/test_Matrix8x6_WithAllRows.cpp
  spmv_formats/test_Matrix8x6_WithEmptyRows.cpp
  spmv_formats/test_PartMatrix5x8_WithAllRows.cpp
  spmv_formats/test_PartMatrix14x5_WithAllRows.cpp
  # Fixed-point Type
  fixed_point/test_fixed_t.cpp
)

target_link_libraries(ANNarchy_CPP_Templates gtest gtest_main gcov)
