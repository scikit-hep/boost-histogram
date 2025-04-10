pybind11_find_import(numpy)
pybind11_find_import(pytest)
pybind11_find_import(pytest-benchmark)

# Much better test target on CMake 3.29+
set(CMAKE_SKIP_TEST_ALL_DEPENDENCY FALSE)

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
    COMMAND ${Python_EXECUTABLE} -m pytest "${TEST_FILE}" --rootdir=.
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
  set_tests_properties(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 5)
endforeach()
