cmake_minimum_required(VERSION 3.12)

project(histogram LANGUAGES CXX VERSION 0.1)

# Boost histogram requires C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adding folders to keep the structure a bit nicer in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# This is a standard recipe for setting a default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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
    extern/core/include
    extern/histogram/include
    extern/mp11/include
    extern/throw_exception/include
    )

# This makes IDE's like XCode mimic the Boost Histogram structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/extern/histogram/include PREFIX "Header Files" FILES ${BOOST_HIST_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${BOOST_HIST_PY_HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${BOOST_HIST_PY_SRC})

# Cause warnings to be errors (not recommended for MSVC, since PyBind11 might cause a few there)
option(BOOST_HISTOGRAM_ERRORS "Make warnings errors (for CI mostly)")

# Adding warnings
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_compile_options(histogram PRIVATE -Wall -Wextra -pedantic-errors -Wconversion -Wsign-conversion)
    if(BOOST_HISTOGRAM_ERRORS)
        target_compile_options(histogram PRIVATE -Werror)
    endif()

    # Python 2 will not be compatible with C++17, but let's not be reminded of that over and over again here.
    if("${PYTHON_VERSION_MAJOR}" LESS 3)
         target_compile_options(histogram PRIVATE -Wno-deprecated-register)
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    target_compile_options(histogram PRIVATE /W4)
    if(BOOST_HISTOGRAM_ERRORS)
        target_compile_options(histogram PRIVATE /WE)
    endif()
endif()



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
configure_file(
    "boost/histogram_utils.py"
    "${CMAKE_CURRENT_BINARY_DIR}/boost/histogram_utils.py"
    COPYONLY)
configure_file(
    "boost/histogram_version.py"
    "${CMAKE_CURRENT_BINARY_DIR}/boost/histogram_version.py"
    COPYONLY)


# Tests (Requires pytest to be available to run)
include(CTest)

function(python_import PACKAGE_NAME)
    string(REPLACE "-" "_" PYTHON_PACKAGE_NAME "${PACKAGE_NAME}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import ${PYTHON_PACKAGE_NAME}" RESULT_VARIABLE RESULT_PRESENT)
    if(RESULT_PRESENT)
        message(SEND_ERROR "You should pip install ${PACKAGE_NAME}")
    endif()
endfunction()

if(BUILD_TESTING)

    python_import(numpy)
    python_import(pytest)
    python_import(pytest-benchmark)

    # Support for running from build directory
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pytest.ini"
        "[pytest]\n"
        "addopts = --benchmark-disable\n"
        "testpaths = . ../tests\n"
        )

    # Support plain "pytest" in addition to "python -m pytest"
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/conftest.py"
        "import os, sys\n"
        "sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))\n"
        )


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



