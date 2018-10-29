
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(kernel_launcher ${sources})
target_include_directories(kernel_launcher PUBLIC ${HSA_HEADER_DIR})
#target_include_directories(kernel_launcher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernel_launcher ${HSA_LIBRARY})

add_subdirectory(kernel)