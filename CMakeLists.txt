cmake_minimum_required(VERSION 3.10...3.18)

project(BOOST_HISTOGRAM LANGUAGES CXX)
# Version is added later

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
set(BH_RULE_LAUNCH_COMPILE
    ""
    CACHE STRING [=[Use a command, like "time" to wrap the compile]=])
if(NOT BH_RULE_LAUNCH_COMPILE STREQUAL "")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${BH_RULE_LAUNCH_COMPILE}")
endif()
set(BH_RULE_LAUNCH_LINK
    ""
    CACHE STRING [=[Use a command, like "time" to wrap the link]=])
if(NOT BH_RULE_LAUNCH_LINK STREQUAL "")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${BH_RULE_LAUNCH_LINK}")
endif()

# This is a standard recipe for setting a default build type
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Display versions
message(STATUS "CMake ${CMAKE_VERSION}")

# Adding pybind11 and setting up Python
# Will display pybind11 version
add_subdirectory(extern/pybind11)

message(STATUS "Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# This is completely optional and just adds hints to IDEs - no affect on build at all.
file(GLOB_RECURSE BOOST_HIST_FILES "extern/histogram/include/*.hpp")
file(GLOB_RECURSE BOOST_HIST_PY_HEADERS "include/bh_python/*.hpp")

# List the source files for the Python extension
# On some backends (like make), this will regenerate if a file was added/removed
file(GLOB BOOST_HIST_PY_SRC CONFIGURE_DEPENDS src/*.cpp)

# This is the Python module
pybind11_add_module(_core SHARED ${BOOST_HIST_PY_HEADERS} ${BOOST_HIST_PY_SRC} ${BOOST_HIST_FILES})

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

# Useful flags
target_compile_options(_core
                       PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:fast,-funsafe-math-optimizations>)

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

# Collect all the python files and symlink them (3.14+) or copy them (3.12-3.13)
# into the build directory
# Protects from in-source builds (don't do this, please)
if(NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  file(
    GLOB_RECURSE BOOST_HIST_PY_FILES
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    CONFIGURE_DEPENDS "src/boost_histogram/*.py")
  foreach(F IN LISTS BOOST_HIST_PY_FILES)
    if(CMAKE_VERSION VERSION_LESS 3.14)
      get_filename_component(FDIR "${F}" DIRECTORY)
      file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/${F}"
           DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${FDIR}")
    else()
      file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${F}")
      file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/src/${F}" "${CMAKE_CURRENT_BINARY_DIR}/${F}"
           COPY_ON_ERROR SYMBOLIC)
    endif()
  endforeach()
endif()

# Support installing
install(DIRECTORY "src/boost_histogram" DESTINATION ".")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/boost_histogram/version.py"
        DESTINATION "boost_histogram")
install(TARGETS _core DESTINATION "boost_histogram")

# Tests (Requires pytest to be available to run)
include(CTest)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/boost_histogram/version.py")
  set(VERSION_REGEX [=[version[ \t]*=[ \t]*["']([0-9]+\.[0-9]+\.[0-9]+)]=])

  # Read in the line containing the version
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/boost_histogram/version.py" VERSION_STRING
       REGEX [=[version[ \t]*=]=])

  # Pick out just the version
  string(REGEX MATCH [=[[0-9]+\.[0-9]+\.[0-9]+]=] VERSION_STRING "${VERSION_STRING}")
else()
  pybind11_find_import(setuptools_scm REQUIRED)
  pybind11_find_import(toml REQUIRED)

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c
      "from setuptools_scm import get_version; print(get_version(root='${CMAKE_CURRENT_SOURCE_DIR}'))"
    RESULT_VARIABLE VERSION_PRESENT
    OUTPUT_VARIABLE VERSION_FULL_STRING
    ERROR_VARIABLE VERSION_ERROR)

  if(ERROR_VARIABLE)
    message(STATUS "Warning from setuptools_scm:\n${ERROR_VARIABLE}")
  endif()

  if(NOT VERSION_PRESENT EQUAL 0)
    message(
      FATAL_ERROR
        "Cannot detect ${VERSION_PRESENT} the version from setuptools_scm\n${VERSION_STRING}")
  endif()

  string(STRIP "${VERSION_FULL_STRING}" VERSION_FULL_STRING)
  string(REGEX MATCH [=[^[0-9]+\.[0-9]+\.[0-9]+]=] VERSION_STRING "${VERSION_FULL_STRING}")
  string(REPLACE "-" "." VERSION_STRING "${VERSION_STRING}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/boost_histogram/version.py"
       "version = '${VERSION_FULL_STRING}'")
  message(STATUS "Full version output: ${VERSION_FULL_STRING}")
endif()

project(
  BOOST_HISTOGRAM
  LANGUAGES CXX
  VERSION ${VERSION_STRING})
message(STATUS "boost-histogram ${BOOST_HISTOGRAM_VERSION}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
