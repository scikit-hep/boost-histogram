pybind11_find_import(numpy)
pybind11_find_import(pytest)
pybind11_find_import(pytest-benchmark)
if(PYTHON_VERSION VERSION_LESS 3.5)
  pybind11_find_import(typing)
endif()

# Support for running from build directory
file(WRITE "${PROJECT_BINARY_DIR}/pytest.ini" "[pytest]\n" "addopts = --benchmark-disable\n"
                                              "testpaths = ${CMAKE_CURRENT_SOURCE_DIR}\n")

# Support plain "pytest" in addition to "python -m pytest"
file(WRITE "${PROJECT_BINARY_DIR}/conftest.py" "import sys\n"
                                               "sys.path.insert(0, '${PROJECT_BINARY_DIR}')\n")

# Look for all the tests
file(GLOB BOOST_HIST_PY_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")

# Add each test
foreach(TEST_FILE IN LISTS BOOST_HIST_PY_TESTS)
  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  add_test(
    NAME ${TEST_NAME}
    COMMAND ${PYTHON_EXECUTABLE} -m pytest "${TEST_FILE}" --rootdir=.
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
endforeach()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
