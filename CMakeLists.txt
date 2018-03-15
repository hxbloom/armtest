# CMake file of the arm test project



# 
# definitions if compiled on android
# Several lins here were copied form ncnn :
# 
if(CMAKE_TOOLCHAIN_FILE)
  if(ARM_ARCH STREQUAL "aarch32")
  set(ANDROID_TOOLCHAIN clang)
  set(ANDROID_ABI "armeabi-v7a")
  set(ANDROID_ARM_NEON ON)
  set(ANDROID_PLATFORM android-14)
  endif()
  if(ARM_ARCH STREQUAL "aarch64")
  set(ANDROID_TOOLCHAIN clang)
  set(ANDROID_ABI "arm64-v8a")
  set(ANDROID_ARM_NEON ON)
  set(ANDROID_PLATFORM android-21)
  endif()
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()



#
# Version Requirements and project name
#
cmake_minimum_required (VERSION 3.0)
project (armtest)
enable_testing()



#
# submodule: google test
#
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest/googletest")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARIES gtest gtest_main)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)

add_subdirectory(${GTEST_ROOT})
include_directories(${GTEST_ROOT} ${GTEST_INCLUDE_DIR})
find_package(GTest REQUIRED)



#
# compile flags
#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")



#
# execution
# main
#
add_executable(current src/conv77_arm.cpp)
#add_executable(current src/newtest.cpp)
set_property(TARGET current PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET current PROPERTY LINK_FLAGS "-pie")
# flag "-save-temps" is used to generate assembly code
# unblock to generate the assembly code
set_property(TARGET current PROPERTY COMPILE_FLAGS "-save-temps")



#
# execution
# gtest for aarch64
#
if(ARM_ARCH STREQUAL "aarch64")
  file( GLOB AARCH64_GTEST_SOURCE
    tests/aarch64/*.cpp
  )
  add_executable(aarch64_test ${AARCH64_GTEST_SOURCE})
  target_link_libraries (aarch64_test gtest gtest_main)
  add_test(aarch64_test aarch64_test)
endif()


