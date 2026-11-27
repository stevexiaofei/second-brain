cmake_minimum_required(VERSION 3.20)
project(mini_cephfs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_cephfs INTERFACE)
target_include_directories(mini_cephfs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(cephfs_demo src/main.cpp)
add_executable(cephfs_tests tests/tests.cpp)
foreach(target IN ITEMS cephfs_demo cephfs_tests)
  target_link_libraries(${target} PRIVATE mini_cephfs)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /utf-8)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()

enable_testing()
add_test(NAME cephfs_tests COMMAND cephfs_tests)
