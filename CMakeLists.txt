cmake_minimum_required(VERSION 3.12)

project(histogram LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MIN_BOOST_VERSION 1.66)

# setup build
option(BUILD_SERIALIZATION "Build serialization code" ON)
mark_as_advanced(BUILD_SERIALIZATION)

# set special compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  add_compile_options(-D_SCL_SECURE_NO_WARNINGS)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6)
    # -fpermissive needed for cpp_int in gcc-6
    add_compile_options(-fpermissive)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -D__STRICT_ANSI__)
endif()

find_package(Boost ${MIN_BOOST_VERSION} REQUIRED COMPONENTS iostreams serialization)

add_subdirectory(extern/pybind11)

# This is completely optional and just adds hints to IDEs.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")

pybind11_add_module(histogram SHARED
    src/getpython.hpp
    src/module.cpp
    src/axis.cpp
    src/histogram.cpp
    ${BOOST_HIST_FILES}
)

#pybind11_add_module(histogramboost SHARED
#    src/python/axis.cpp
#    src/python/histogram.cpp
#    src/python/module.cpp
#    src/python/serialization_suite.hpp
#    src/python/utility.hpp
#)

target_link_libraries(histogram PUBLIC Boost::boost Boost::iostreams Boost::serialization)
target_include_directories(histogram PUBLIC extern/histogram/include)

#target_link_libraries(histogramboost PUBLIC Boost::boost Boost::iostreams Boost::serialization)
#target_include_directories(histogramboost PUBLIC extern/histogram/include)


# tests
enable_testing()

file(GLOB TEST_SOURCES test/*_test.py examples/*.py ${CMAKE_BINARY_DIR}/doc_test/*.py)
foreach(SRC IN ITEMS ${TEST_SOURCES})
if(SRC MATCHES "/([_a-zA-Z0-9]+)\\.py")
  set(TESTNAME ${CMAKE_MATCH_1})
  file(READ ${SRC} SRC_CONTENT)
  add_test(${TESTNAME} ${PYTHON_EXECUTABLE} ${SRC})
endif()
endforeach()


