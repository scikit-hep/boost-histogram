cmake_minimum_required(VERSION 3.12)

project(histogram LANGUAGES CXX VERSION 0.1)

# Boost histogram requires C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adding folders to keep the structure a bit nicer in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Adding PyBind11 and setting up Python
add_subdirectory(extern/pybind11)

# This is completely optional and just adds hints to IDEs - no affect on build at all.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")

# This is the Python module
pybind11_add_module(boosthistogram SHARED
    src/getpython.hpp
    src/axis.hpp
    src/module.cpp
    src/axis.cpp
    src/histogram.cpp
    src/storage.cpp
    ${BOOST_HIST_FILES}
)

# This makes IDE's like XCode mimic the Boost Histogram structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/extern/histogram/include PREFIX "Header Files" FILES ${BOOST_HIST_FILES})

# These are the Boost header-only libraries required by Boost::Histogram
target_include_directories(boosthistogram PUBLIC
    extern/assert/include
    extern/callable_traits/include
    extern/config/include
    extern/container_hash/include
    extern/core/include
    extern/detail/include
    extern/histogram/include
    extern/integer/include
    extern/iterator/include
    extern/move/include
    extern/mp11/include
    extern/mpl/include
    extern/preprocessor/include
    extern/pybind11/include
    extern/static_assert/include
    extern/throw_exception/include
    extern/type_index/include
    extern/type_traits/include
    extern/utility/include
    extern/variant/include
    )


# Tests (Require pytest to be available)
include(CTest)

add_test(basic_import ${PYTHON_EXECUTABLE} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basic_import.py")



