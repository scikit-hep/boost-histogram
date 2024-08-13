cmake_minimum_required(VERSION 3.24...3.29)

project(BOOST_HISTOGRAM LANGUAGES CXX)
# Version is added later

include(FetchContent)

# Boost histogram requires C++14
set(CMAKE_CXX_STANDARD
    14
    CACHE STRING "The C++ standard to compile with, 14+")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adding folders to keep the structure a bit nicer in IDE's
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# This will force color output at build time if this env variable is set _at
# configure time_. This is useful for CI.
if($ENV{FORCE_COLOR})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

# This is a standard recipe for setting a default build type
set(_default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${_default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${_default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Adding pybind11 and setting up Python
# Will display pybind11 version
set(PYBIND11_FINDPYTHON TRUE)
set(Python_ARTIFACTS_INTERACTIVE TRUE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG bd67643652d3800837f1f41549a2a5adbaa3fafe # v2.13.3
  FIND_PACKAGE_ARGS NAMES pybind11)
FetchContent_MakeAvailable(pybind11)

# Display versions
message(STATUS "CMake ${CMAKE_VERSION}")
message(STATUS "Python ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
if(DEFINED pybind11_VERSION)
  message(STATUS "pybind11 ${pybind11_VERSION}")
endif()

# This is completely optional and just adds hints to IDEs - no affect on build at all.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")
file(GLOB_RECURSE BOOST_HIST_PY_HEADERS "include/bh_python/*.hpp")

# List the source files for the Python extension
# On some backends (like make), this will regenerate if a file was added/removed
file(GLOB BOOST_HIST_PY_SRC CONFIGURE_DEPENDS src/*.cpp)

# This is the Python module
pybind11_add_module(_core MODULE ${BOOST_HIST_PY_HEADERS} ${BOOST_HIST_PY_SRC} ${BOOST_HIST_FILES})

# Add the include directory for boost/histogram/python
target_include_directories(_core PRIVATE include)

# These are the Boost header-only libraries required by Boost::Histogram
target_include_directories(
  _core SYSTEM
  PUBLIC extern/assert/include
         extern/config/include
         extern/core/include
         extern/histogram/include
         extern/mp11/include
         extern/throw_exception/include
         extern/variant2/include)

# Support older Windows from newer MSVC
target_compile_options(_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/d2FH4->)

# Better error messages
target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

# Atomic is required for armv7l
if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
  target_link_libraries(_core PRIVATE atomic)
endif()

# This makes IDE's like XCode mimic the Boost Histogram structure
source_group(
  TREE ${CMAKE_CURRENT_SOURCE_DIR}/extern/histogram/include
  PREFIX "Header Files"
  FILES ${BOOST_HIST_FILES})
source_group(
  TREE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PREFIX "Header Files"
  FILES ${BOOST_HIST_PY_HEADERS})
source_group(
  TREE ${CMAKE_CURRENT_SOURCE_DIR}/src
  PREFIX "Source Files"
  FILES ${BOOST_HIST_PY_SRC})

# Cause warnings to be errors (not recommended for MSVC, since pybind11 might cause a few there)
option(BOOST_HISTOGRAM_ERRORS "Make warnings errors (for CI mostly)")

# Adding warnings
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  target_compile_options(
    _core
    PRIVATE -Wall
            -Wextra
            -pedantic-errors
            -Wconversion
            -Wsign-conversion
            -Wsign-compare
            -Wno-unused-value)
  if(BOOST_HISTOGRAM_ERRORS)
    target_compile_options(_core PRIVATE -Werror)
  endif()

elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
  target_compile_options(_core PRIVATE /W4)
  if(BOOST_HISTOGRAM_ERRORS)
    target_compile_options(_core PRIVATE /WE)
  endif()
endif()

# This allows setting the maximum number of axes in a histogram
set(BOOST_HISTOGRAM_DETAIL_AXES_LIMIT
    ""
    CACHE STRING "Set the maximum number of axis in a histogram (affects compile time and size)")
if(NOT "${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT}" STREQUAL "")
  target_compile_definitions(
    _core PRIVATE BOOST_HISTOGRAM_DETAIL_AXES_LIMIT=${BOOST_HISTOGRAM_DETAIL_AXES_LIMIT})
endif()

# Make the output library be in boost_histogram/_core...
# This is a generator expression to avoid Debug/Release subdirectories in IDEs,
# which confuses Python.
set_property(TARGET _core PROPERTY LIBRARY_OUTPUT_DIRECTORY "$<1:boost_histogram>")

# Collect all the python files and symlink them into the build directory
# Protects from in-source builds (don't do this, please)
if(NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" AND NOT DEFINED
                                                                                SKBUILD)
  file(
    GLOB_RECURSE BOOST_HIST_PY_FILES
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    CONFIGURE_DEPENDS "src/boost_histogram/*.py")
  foreach(F IN LISTS BOOST_HIST_PY_FILES)
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${F}")
    file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/src/${F}" "${CMAKE_CURRENT_BINARY_DIR}/${F}"
         COPY_ON_ERROR SYMBOLIC)
  endforeach()
endif()

# Support installing
install(TARGETS _core DESTINATION "boost_histogram")

if(NOT DEFINED SKBUILD)
  install(DIRECTORY "src/boost_histogram" DESTINATION ".")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/boost_histogram/version.py"
          DESTINATION "boost_histogram")

  # Tests (Requires pytest to be available to run)
  include(CTest)
endif()

if(DEFINED SKBUILD)
  # Don't worry about the version
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/boost_histogram/version.py")
  set(VERSION_REGEX [=[version[ \t]*=[ \t]*["']([0-9]+\.[0-9]+\.[0-9]+)]=])

  # Read in the line containing the version
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/boost_histogram/version.py" VERSION_STRING
       REGEX [=[version[ \t]*=]=])

  # Pick out just the version
  string(REGEX MATCH [=[[0-9]+\.[0-9]+\.[0-9]+]=] VERSION_STRING "${VERSION_STRING}")
else()
  pybind11_find_import(setuptools_scm REQUIRED)

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "from setuptools_scm import get_version; print(get_version())"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VERSION_FULL_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)

  string(REGEX MATCH [=[^[0-9]+\.[0-9]+\.[0-9]+]=] VERSION_STRING "${VERSION_FULL_STRING}")
  string(REPLACE "-" "." VERSION_STRING "${VERSION_STRING}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/boost_histogram/version.py"
       "version = '${VERSION_FULL_STRING}'")
  message(STATUS "Full version output: ${VERSION_FULL_STRING}")
endif()

if(NOT DEFINED SKBUILD)
  project(
    BOOST_HISTOGRAM
    LANGUAGES CXX
    VERSION ${VERSION_STRING})
  message(STATUS "boost-histogram ${BOOST_HISTOGRAM_VERSION}")

  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()
