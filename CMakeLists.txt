cmake_minimum_required(VERSION 3.12)

project(histogram LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MIN_BOOST_VERSION 1.66)

find_package(Boost ${MIN_BOOST_VERSION} REQUIRED COMPONENTS iostreams serialization)

add_subdirectory(extern/pybind11)

# This is completely optional and just adds hints to IDEs.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")

pybind11_add_module(hist SHARED
    src/getpython.hpp
    src/module.cpp
    src/axis.cpp
    src/histogram.cpp
    ${BOOST_HIST_FILES}
)

target_link_libraries(hist PUBLIC Boost::boost Boost::iostreams Boost::serialization)
target_include_directories(hist PUBLIC extern/histogram/include)

# tests
include(CTest)

file(GLOB TEST_SOURCES test/*_test.py examples/*.py ${CMAKE_BINARY_DIR}/doc_test/*.py)
foreach(SRC IN ITEMS ${TEST_SOURCES})
if(SRC MATCHES "/([_a-zA-Z0-9]+)\\.py")
  set(TESTNAME ${CMAKE_MATCH_1})
  file(READ ${SRC} SRC_CONTENT)
  add_test(${TESTNAME} ${PYTHON_EXECUTABLE} ${SRC})
endif()
endforeach()


