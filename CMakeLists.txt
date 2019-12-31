cmake_minimum_required(VERSION 3.10...3.15)

# Grab the version from the python files
set(VERSION_REGEX [=[__version__[ \t]*=[ \t]*"(.+)"]=])

# Read in the line containing the version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/boost_histogram/version.py"
    VERSION_STRING REGEX ${VERSION_REGEX})

# Pick out just the version
string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION_STRING "${VERSION_STRING}")

project(BOOST_HISTOGRAM LANGUAGES CXX VERSION ${VERSION_STRING})


# Boost histogram requires C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adding folders to keep the structure a bit nicer in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Make time/memory taken to compile/link available as a setting
#
# Useful settings:
# Linux:
#   "time"
#   "time -v"
#   "time -f '%U user %S system %E elapsed %P CPU %M KB'"
# macOS:
#   "time -l"
# macOS with brew install gnu-time:
#   "gtime -f '%U user %S system %E elapsed %P CPU %M KB'"
#
set(BH_RULE_LAUNCH_COMPILE "" CACHE STRING [=[Use a command, like "time" to wrap the compile]=])
if(NOT BH_RULE_LAUNCH_COMPILE STREQUAL "")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${BH_RULE_LAUNCH_COMPILE}")
endif()
set(BH_RULE_LAUNCH_LINK "" CACHE STRING [=[Use a command, like "time" to wrap the link]=])
if(NOT BH_RULE_LAUNCH_LINK STREQUAL "")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${BH_RULE_LAUNCH_LINK}")
endif()

# This is a standard recipe for setting a default build type
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Display versions
message(STATUS "boost-histogram ${BOOST_HISTOGRAM_VERSION}")
message(STATUS "CMake ${CMAKE_VERSION}")

# Adding PyBind11 and setting up Python
# Will display PyBind11 version
add_subdirectory(extern/pybind11)

message(STATUS "Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# This is completely optional and just adds hints to IDEs - no affect on build at all.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")
file(GLOB_RECURSE BOOST_HIST_PY_HEADERS "include/bh_python/*.hpp")

# List the source files for the Python extension
# On some backends (like make), this will regenerate if a file was added/removed
file(GLOB
    BOOST_HIST_PY_SRC
    CONFIGURE_DEPENDS
    src/*.cpp
    )


# This is the Python module
pybind11_add_module(_core SHARED
    ${BOOST_HIST_PY_HEADERS}
    ${BOOST_HIST_PY_SRC}
    ${BOOST_HIST_FILES}
)

# Add the include directory for boost/histogram/python
target_include_directories(_core PRIVATE include)

# These are the Boost header-only libraries required by Boost::Histogram
target_include_directories(_core SYSTEM PUBLIC
    extern/assert/include
    extern/config/include
    extern/core/include
    extern/histogram/include
    extern/mp11/include
    extern/throw_exception/include
    extern/variant2/include
    )

# Useful flags
target_compile_options(_core
  PRIVATE
    $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:fast,-funsafe-math-optimizations>
)

# This makes IDE's like XCode mimic the Boost Histogram structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/extern/histogram/include PREFIX "Header Files" FILES ${BOOST_HIST_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${BOOST_HIST_PY_HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${BOOST_HIST_PY_SRC})

# Cause warnings to be errors (not recommended for MSVC, since PyBind11 might cause a few there)
option(BOOST_HISTOGRAM_ERRORS "Make warnings errors (for CI mostly)")

# Adding warnings
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_compile_options(_core PRIVATE -Wall -Wextra -pedantic-errors -Wconversion -Wsign-conversion -Wsign-compare)
    if(BOOST_HISTOGRAM_ERRORS)
        target_compile_options(_core PRIVATE -Werror)
    endif()

    # Python 2 will not be compatible with C++17, but let's not be reminded of that over and over again here.
    if("${PYTHON_VERSION_MAJOR}" LESS 3 AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
         target_compile_options(_core PRIVATE -Wno-deprecated-register)
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    target_compile_options(_core PRIVATE /W4)
    if(BOOST_HISTOGRAM_ERRORS)
        target_compile_options(_core PRIVATE /WE)
    endif()
endif()

# This allows setting the maximum number of axes in a histogram
set(BOOST_HISTOGRAM_DETAIL_AXES_LIMIT "" CACHE STRING "Set the maximum number of axis in a histogram (affects compile time and size)")
if(NOT "${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT}" STREQUAL "")
    target_compile_definitions(_core PRIVATE BOOST_HISTOGRAM_DETAIL_AXES_LIMIT=${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT})
endif()

# Make the output library be in boost_histogram/_core...
# This is a generator expression to avoid Debug/Release subdirectories in IDEs,
# which confuses Python.
set_property(TARGET _core PROPERTY LIBRARY_OUTPUT_DIRECTORY "$<1:boost_histogram>")

# Collect all the python files and symlink them (3.14+) or copy them (3.12-3.13)
# into the build directory
# Protects from in-source builds (don't do this, please)
if(NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    file(GLOB_RECURSE
        BOOST_HIST_PY_FILES
        LIST_DIRECTORIES false
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        CONFIGURE_DEPENDS
        "boost_histogram/*.py")
    foreach(F IN LISTS BOOST_HIST_PY_FILES)
        if(CMAKE_VERSION VERSION_LESS 3.14)
            get_filename_component(FDIR "${F}" DIRECTORY )
            file(COPY       "${CMAKE_CURRENT_SOURCE_DIR}/${F}"
                DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${FDIR}")
        else()
            file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${F}")
            file(CREATE_LINK
                "${CMAKE_CURRENT_SOURCE_DIR}/${F}"
                "${CMAKE_CURRENT_BINARY_DIR}/${F}"
                COPY_ON_ERROR SYMBOLIC
                )
        endif()
    endforeach()
endif()

# Support installing
install(DIRECTORY "boost_histogram" DESTINATION ".")
install(TARGETS _core DESTINATION "boost_histogram")


# Tests (Requires pytest to be available to run)
include(CTest)

# Check to see if modules are importable.
function(python_import)
    foreach(PACKAGE_NAME IN LISTS ARGN)
        string(REPLACE "-" "_" PYTHON_PACKAGE_NAME "${PACKAGE_NAME}")
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import ${PYTHON_PACKAGE_NAME}"
            RESULT_VARIABLE RESULT_PRESENT
            OUTPUT_QUIET ERROR_QUIET)
        if(RESULT_PRESENT)
            list(APPEND FAILED_NAMES "${PACKAGE_NAME}")
        endif()
    endforeach()
    if(FAILED_NAMES)
        list(JOIN FAILED_NAMES " " FAILED_NAMES)
        message(SEND_ERROR "Dependencies missing, try installing:\n  ${PYTHON_EXECUTABLE} -m pip install ${FAILED_NAMES}\n")
    endif()
endfunction()

if(BUILD_TESTING)

    python_import(numpy pytest pytest-benchmark)

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
