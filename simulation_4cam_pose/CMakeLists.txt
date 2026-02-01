cmake_minimum_required(VERSION 3.12)
project(tri_pattern_sim_sequence CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ros)
set(ROS_NATIVE_INC_DIRS ros/include
        ros/console_bridge/include
        ros/roscpp_core/cpp_common/include
        ros/roscpp_core/rostime/include
        ros/roscpp_core/roscpp_serialization/include
        ros/roscpp_core/roscpp_traits/include
        ros/ros_comm/utilities/roslz4/include
        ros/ros_comm/tools/rosbag_storage/include
        ros/cv_bridge/include
        ros/cv_bridge/src
        ros/cv_bridge/include
)

add_executable(simulation_4cam_pose main_smooth_fixed.cpp)
target_compile_definitions(simulation_4cam_pose PRIVATE -D_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
target_compile_options(simulation_4cam_pose PRIVATE -O3 -w)
target_include_directories(simulation_4cam_pose PRIVATE ${ROS_NATIVE_INC_DIRS})
target_link_libraries(simulation_4cam_pose libcvbridge rosbag)