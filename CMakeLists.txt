cmake_minimum_required(VERSION 2.7)
project(kernel_launcher_amdgpu)

# find llvm
find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

find_program(CLANG NAMES clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if (CLANG)
    message(STATUS "Found Clang: ${CLANG}")
else()
    message(STATUS "Clang not found")
endif()

# path
if(NOT HSA_ROOT)
    set(HSA_ROOT "/opt/rocm/hsa/" CACHE PATH "HSA runtime path")
endif()
set(HSA_HEADER_DIR "${HSA_ROOT}/include/hsa" CACHE PATH "HSA include path")
if(NOT ROCM_ROOT)
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm runtime path")
endif()
set(ROCM_HEADER_DIR ${ROCM_ROOT}/include CACHE PATH "ROCm include path")

# arch
if(NOT GPU_ARCH)
    set(GPU_ARCH "gfx900" CACHE STRING "gpu arch")
endif()

# validating path
find_path(HSA_HEADER hsa.h PATHS ${ROCM_HEADER_DIR} ${HSA_HEADER_DIR} NO_DEFAULT_PATH)
find_path(HSA_HEADER hsa.h)
if(NOT HSA_HEADER)
    message(STATUS "HSA header not found. Use -DHSA_HEADER_DIR=<path_to_hsa.h>.")
endif()

find_library(HSA_LIBRARY hsa-runtime64 PATHS ${HSA_LIBRARY_DIR} ${ROCM_ROOT}/lib ${HSA_ROOT}/lib NO_DEFAULT_PATH)
find_library(HSA_LIBRARY hsa-runtime64)
if(HSA_LIBRARY)
    message(STATUS "Found HSA library: ${HSA_LIBRARY}")
else()
    message(STATUS "HSA runtime library not found. Use -DHSA_LIBRARY_DIR=<path_to_libhsa-runtime64.so>.")
endif()

macro(asm_assemble f)
add_custom_command(
  OUTPUT ${f}.o
  COMMAND
    ${CLANG}  -x assembler -target amdgcn-amd-amdhsa--${GPU_ARCH} -mcpu=${GPU_ARCH} -c -o ${f}.o ${CMAKE_CURRENT_SOURCE_DIR}/${f}.s
  DEPENDS ${f}.s
  COMMENT "Assembling ${f}.s to ${f}.o"
)
endmacro(asm_assemble)

macro(asm_link f)
add_custom_command(
  OUTPUT ${f}.co
  COMMAND
    ${CLANG} -target amdgcn--amdhsa ${f}.o -o ${f}.co
  DEPENDS ${f}.o
  COMMENT "Linking ${f}.o to ${f}.co with clang"
)
endmacro(asm_link)

# single file example
macro(asm_example f)
  asm_assemble(${f})
  asm_link(${f})
  add_custom_target(${f}_co ALL DEPENDS ${f}.co)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${f}.co DESTINATION bin COMPONENT examples)
endmacro(asm_example)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

add_subdirectory(src)

