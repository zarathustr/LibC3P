cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wno-deprecated-declarations")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-catch-value")
endif()

#find_package(catkin REQUIRED COMPONENTSe cpp_common roscpp_serialization roscpp_traits rostime roslz4)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
find_package(BZip2 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED
        COMPONENTS
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        CONFIG
)

pkg_check_modules(LZ4LIB REQUIRED liblz4)

file(GLOB CONSOLE_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/console_bridge/src/*.cpp")
file(GLOB CPP_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/cpp_common/src/*.cpp")
file(GLOB ROSCPP_SERIALIZATION_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/roscpp_serialization/src/*.cpp")
file(GLOB ROSTIME_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/rostime/src/*.cpp")

file(GLOB ROSBAG_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/tools/rosbag_storage/src/*.cpp")
file(GLOB ROSLZ4_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/utilities/roslz4/src/[a-z]*.c")

message("roslz4: ${ROSLZ4_SRCS}")
add_library(rosbag STATIC ${ROSBAG_SRCS} ${ROSTIME_SRCS} ${CPP_COMMON_SRCS} ${ROSCPP_SERIALIZATION_SRCS} ${ROSLZ4_SRCS} ${CONSOLE_BRIDGE_SRCS})

set(ROS_NATIVE_INC_DIRS include
        console_bridge/include
        roscpp_core/cpp_common/include
        roscpp_core/rostime/include
        roscpp_core/roscpp_serialization/include
        roscpp_core/roscpp_traits/include
        ros_comm/utilities/roslz4/include
        ros_comm/tools/rosbag_storage/include
        cv_bridge/include
        cv_bridge/src
        ${Boost_INCLUDE_DIRS}
        ${LZ4LIB_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

target_include_directories(rosbag PUBLIC ${ROS_NATIVE_INC_DIRS})
target_link_libraries(rosbag -L/usr/local/lib ${LZ4LIB_LIBRARIES} ${BZIP2_LIBRARIES} -llz4 boost_system)

add_library(libcvbridge cv_bridge/src/cv_bridge.cpp cv_bridge/src/rgb_colors.cpp)

target_include_directories(libcvbridge PUBLIC ${ROS_NATIVE_INC_DIRS})
target_link_libraries(libcvbridge ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
if(OpenCV_FOUND)
    if("${OpenCV_VERSION}" GREATER_EQUAL "4")
        target_compile_definitions(libcvbridge PRIVATE -DUSE_OPENCV4)
    elseif("${OpenCV_VERSION}" GREATER_EQUAL "3")
        target_compile_definitions(libcvbridge PRIVATE -DUSE_OPENCV3)
    else()
        message(FATAL_ERROR "OpenCV Version: ${OPENCV_VERSION} is too old for this project!")
    endif()
endif()
