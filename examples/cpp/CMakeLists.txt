# This file lets you quickly build projects with the exact same boost-histogram
# settings that the Python bindings use.

cmake_minimum_required(VERSION 3.14...3.20)

project(BOOST_HISTOGRAM_CPP LANGUAGES CXX)

add_library(boost_histogram_cpp IMPORTED INTERFACE)

# these are the boost header-only libraries required by boost::histogram
target_include_directories(
  boost_histogram_cpp
  INTERFACE ../../extern/assert/include
            ../../extern/config/include
            ../../extern/core/include
            ../../extern/histogram/include
            ../../extern/mp11/include
            ../../extern/throw_exception/include
            ../../extern/variant2/include)

target_compile_options(
  boost_histogram_cpp INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:fast,-funsafe-math-optimizations>)

target_compile_features(boost_histogram_cpp INTERFACE cxx_std_14)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
  add_executable(simple simple.cpp)
  target_link_libraries(simple PRIVATE boost_histogram_cpp)
endif()
