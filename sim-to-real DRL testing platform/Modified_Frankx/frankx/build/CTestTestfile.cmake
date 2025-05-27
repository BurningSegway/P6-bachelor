# CMake generated Testfile for 
# Source directory: /home/morten/frankx
# Build directory: /home/morten/frankx/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(kinematics-test "/home/morten/frankx/build/kinematics-test")
set_tests_properties(kinematics-test PROPERTIES  _BACKTRACE_TRIPLES "/home/morten/frankx/CMakeLists.txt;84;add_test;/home/morten/frankx/CMakeLists.txt;0;")
add_test(unit-test "/home/morten/frankx/build/unit-test")
set_tests_properties(unit-test PROPERTIES  _BACKTRACE_TRIPLES "/home/morten/frankx/CMakeLists.txt;84;add_test;/home/morten/frankx/CMakeLists.txt;0;")
add_test(path-test "/home/morten/frankx/build/path-test")
set_tests_properties(path-test PROPERTIES  _BACKTRACE_TRIPLES "/home/morten/frankx/CMakeLists.txt;84;add_test;/home/morten/frankx/CMakeLists.txt;0;")
subdirs("affx")
subdirs("ruckig")
