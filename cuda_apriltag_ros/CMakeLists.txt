cmake_minimum_required(VERSION 3.0.2)
project(cuda_apriltag_ros)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  geometry_msgs
  image_geometry
  image_transport
  message_generation
  nodelet
  pluginlib
  roscpp
  sensor_msgs
  std_msgs
  tf
  apriltag_ros
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED)

# CUDA
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# NVAprilTags
include_directories(nvapriltags)
add_library(nvapriltags STATIC IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_x86_64_cuda_11_4/libapril_tagging.a)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a.patched)
endif()

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# We default to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Coverage" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

add_compile_options("-O3" "-funsafe-loop-optimizations" "-fsee" "-funroll-loops" "-fno-math-errno" "-funsafe-math-optimizations" "-ffinite-math-only" "-fno-signed-zeros")

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_52>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=compute_86>")


add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++11>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>")

# Note: These options have been turned off to allow for binary releases - 
# in local builds, they can be reactivated to achieve higher performance.
# if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64" OR "x86_32")
#   message("enabling msse2 for x86_64 or x86_32 architecture")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -msse2 ")
# endif()
# if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "arm")
#   message("enabling -mfpu=neon -mfloat-abi=softfp for ARM architecture")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mcpu=cortex-a9 -mfpu=neon -mtune=cortex-a9 -mvectorize-with-neon-quad -ffast-math ")
# endif()

# Extract the include directories and libraries from apriltag::apriltag as catkin_package() does not support modern cmake.
get_target_property(apriltag_INCLUDE_DIRS apriltag::apriltag INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(apriltag_LIBRARIES apriltag::apriltag INTERFACE_LINK_LIBRARIES)

catkin_package(
  INCLUDE_DIRS
    include
    ${EIGEN3_INCLUDE_DIRS}
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    image_transport
    message_runtime
    nodelet
    pluginlib
    roscpp
    sensor_msgs
    std_msgs
    tf
  DEPENDS
    apriltag
    OpenCV
  LIBRARIES
    ${PROJECT_NAME}_common
    ${PROJECT_NAME}_continuous_detector
    ${PROJECT_NAME}_single_image_detector
)

###########
## Build ##
###########

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_common src/common_functions.cpp)
add_dependencies(${PROJECT_NAME}_common ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_common
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES} 
  ${CUDA_nvToolsExt_LIBRARY}
  ${CUDA_LIBRARIES}
  nvapriltags
  apriltag::apriltag
)

add_library(${PROJECT_NAME}_continuous_detector src/continuous_detector.cpp)
target_link_libraries(${PROJECT_NAME}_continuous_detector ${PROJECT_NAME}_common ${catkin_LIBRARIES})

add_library(${PROJECT_NAME}_single_image_detector src/single_image_detector.cpp)
target_link_libraries(${PROJECT_NAME}_single_image_detector ${PROJECT_NAME}_common ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_continuous_node src/${PROJECT_NAME}_continuous_node.cpp)
add_dependencies(${PROJECT_NAME}_continuous_node ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_continuous_node ${PROJECT_NAME}_continuous_detector ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_single_image_server_node src/${PROJECT_NAME}_single_image_server_node.cpp)
add_dependencies(${PROJECT_NAME}_single_image_server_node ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_single_image_server_node ${PROJECT_NAME}_single_image_detector ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_single_image_client_node src/${PROJECT_NAME}_single_image_client_node.cpp)
add_dependencies(${PROJECT_NAME}_single_image_client_node ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_single_image_client_node ${PROJECT_NAME}_common ${catkin_LIBRARIES})


#############
## Install ##
#############

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS
  ${PROJECT_NAME}_common
  ${PROJECT_NAME}_continuous_detector
  ${PROJECT_NAME}_continuous_node
  ${PROJECT_NAME}_single_image_client_node
  ${PROJECT_NAME}_single_image_detector
  ${PROJECT_NAME}_single_image_server_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(PROGRAMS scripts/analyze_image DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
