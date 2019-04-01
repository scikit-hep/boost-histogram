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
file(GLOB_RECURSE BOOST_HIST_PY_HEADERS "include/*.hpp")

# List the source files for the Python extension
# On some backends (like make), this will regenerate if a file was added/removed
file(GLOB_RECURSE
    BOOST_HIST_PY_SRC
    CONFIGURE_DEPENDS
    src/*.cpp
    )


# This is the Python module
pybind11_add_module(histogram SHARED
    ${BOOST_HIST_PY_HEADERS}
    ${BOOST_HIST_PY_SRC}
    ${BOOST_HIST_FILES}
)

# Add the include directory for boost/histogram/python
target_include_directories(histogram PRIVATE include)

# These are the Boost header-only libraries required by Boost::Histogram
target_include_directories(histogram SYSTEM PUBLIC
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
    extern/static_assert/include
    extern/throw_exception/include
    extern/type_index/include
    extern/type_traits/include
    extern/utility/include
    extern/variant/include
    )

# This is the Cereal heaader-only library required for serialization
target_include_directories(histogram SYSTEM PUBLIC
    extern/cereal/include
    )

# This makes IDE's like XCode mimic the Boost Histogram structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/extern/histogram/include PREFIX "Header Files" FILES ${BOOST_HIST_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${BOOST_HIST_PY_HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${BOOST_HIST_PY_SRC})

# Cause warnings to be errors (not recommended for MSVC, since PyBind11 might cause a few there - add /WX if you want to try)
option(BOOST_HISTOGRAM_ERRORS "Make warnings errors (for CI mostly)")

# Adding warnings
target_compile_options(histogram PRIVATE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                           -Wall -Wextra -pedantic-errors -Wconversion -Wsign-conversion -Wno-return-std-move>
                       $<$<CXX_COMPILER_ID:MSVC>:
                           /W4>)
if(BOOST_HISTOGRAM_ERRORS)
    target_compile_options(histogram PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:
            /WE>)
endif()

target_compile_options(histogram PRIVATE
                       $<$<CXX_COMPILER_ID:AppleClang>:
                           -ftemplate-backtrace-limit=0>)

set(BOOST_HISTOGRAM_DETAIL_AXES_LIMIT "" CACHE STRING "Set the maximum number of axis in a histogram (affects compile time and size)")
if(NOT "${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT}" STREQUAL "")
    target_compile_definitions(histogram PRIVATE BOOST_HISTOGRAM_DETAIL_AXES_LIMIT=${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT})
endif()

# Make the output be in boost/...
set_property(TARGET histogram PROPERTY LIBRARY_OUTPUT_DIRECTORY "boost")
configure_file(
    "boost/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/boost/__init__.py"
    COPYONLY)

# Tests (Requires pytest to be available to run)
include(CTest)

if(BUILD_TESTING)
    # Look for all the tests
    file(GLOB
        BOOST_HIST_PY_TESTS
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py"
        )

    # Add each test
    foreach(TEST_FILE IN LISTS BOOST_HIST_PY_TESTS)
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
        add_test(${TEST_NAME} ${PYTHON_EXECUTABLE} -m pytest "${TEST_FILE}")
    endforeach()
endif()



