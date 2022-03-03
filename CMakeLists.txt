cmake_minimum_required(VERSION 2.8)
project(SSBTree)

if( NOT CMAKE_BUILD_TYPE )
    message(STATUS "No build type selected, default to Release")
    set( CMAKE_BUILD_TYPE Release)
else()
    message(STATUS "Build type is set to ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17   -fPIC -march=native -mrtm -mcx16 -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -Wall -Wextra -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free  -DNDEBUG")

execute_process(COMMAND cat /proc/cpuinfo COMMAND grep clflush OUTPUT_VARIABLE ENABLE_CLFLUSH)
execute_process(COMMAND cat /proc/cpuinfo COMMAND grep clflushopt OUTPUT_VARIABLE ENABLE_CLFLUSHOPT)
execute_process(COMMAND cat /proc/cpuinfo COMMAND grep clwb OUTPUT_VARIABLE ENABLE_CLWB)
execute_process(COMMAND cat /proc/cpuinfo COMMAND grep avx512 OUTPUT_VARIABLE ENABLE_AVX512)


if(ENABLE_CLWB)
    add_definitions(-DCLWB)
    message(STATUS "Looking for clwb instruction - found")
elseif(ENABLE_CLFLUSHOPT)
    add_definitions(-DCLFLUSH_OPT)
    message(STATUS "Looking for clwb instruction - not found")
    message(STATUS "Looking for clflushopt instruction - found")
elseif(ENABLE_CLFLUSH)
    add_definitions(-DCLFLUSH)
    message(STATUS "Looking for clwb instruction - not found")
    message(STATUS "Looking for clflushopt instruction - not found")
    message(STATUS "Looking for clflush instruction - found")
else()
    message(FATAL_ERROR "Cannot find any flush instructions (clflush, clflushopt, clwb)")
endif()

if(ENABLE_AVX512)
    set(CMAKE_CXX_FLAGS "-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd ${CMAKE_CXX_FLAGS}")
    add_definitions(-DUSE_AVX512)
    message(STATUS "Looking for avx512 instruction - found")
else()
    message(STATUS "Looking for avx512 instruction - not found")
endif()

option(REBALANCE, "Merge unuse nodes to rebalance tree." off)
if(${REBALANCE})
  add_definitions(-DREBALANCE)
  message(STATUS "REBALANCE: defined")
else()
  message(STATUS "REBALANCE: not defined")
endif()


find_library(JemallocLib jemalloc)
find_library(TbbLib tbb)
find_library(Pmemobj pmemobj)
find_library(Pmem pmem)

add_library(EPOCHE SHARED Epoche.cpp Epoche.h)

set(INDEX_FILES SSBTree.cpp Epoche.cpp)

add_library(Indexes ${INDEX_FILES})
target_link_libraries(Indexes  ${JemallocLib} ${TbbLib} ${Pmemobj} ${Pmem} )

set(SSBTree_TEST example.cpp)
add_executable(example ${SSBTree_TEST})

target_link_libraries(example Indexes atomic boost_system boost_thread)


add_library(SSBTree_pibench_wrapper SHARED SSBTree_pibench_wrapper.cc)
target_link_libraries(SSBTree_pibench_wrapper Indexes )