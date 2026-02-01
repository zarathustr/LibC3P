cmake_minimum_required(VERSION 3.0.2)
project(aruco_extrinsic_calib_c3p)

# Catkin-based ROS1 package (needed for rosbag + sensor_msgs)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  sensor_msgs
)

# OpenCV 3.4.5 (aruco is in opencv_contrib)
find_package(OpenCV 3.4.5 REQUIRED COMPONENTS
  core imgproc imgcodecs highgui videoio calib3d aruco
)

# YAML parser for calib-camchain.yaml
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME}_core
  src/camchain_yaml.cpp
  src/pose_utils.cpp
  src/handeye_axxb.cpp
  src/xy_yaml.cpp
  src/xyzw_yaml.cpp
)
target_link_libraries(${PROJECT_NAME}_core
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

add_executable(extract_aruco_from_bag
  src/extract_aruco_from_bag.cpp
)
target_link_libraries(extract_aruco_from_bag
  ${PROJECT_NAME}_core
)

add_executable(verify_extrinsics
  src/verify_extrinsics.cpp
)
target_link_libraries(verify_extrinsics
  ${PROJECT_NAME}_core
)

add_executable(robot_world_handeye_solver
  src/robot_world_handeye_solver.cpp
)
target_link_libraries(robot_world_handeye_solver
  ${PROJECT_NAME}_core
)

add_executable(overlay_reprojection_from_bag
  src/overlay_reprojection_from_bag.cpp
)
target_link_libraries(overlay_reprojection_from_bag
  ${PROJECT_NAME}_core
)

add_executable(verify_axby_zcwd
  src/verify_axby_zcwd.cpp
)
target_link_libraries(verify_axby_zcwd
  ${PROJECT_NAME}_core
)

add_executable(overlay_reprojection_zcwd_from_bag
  src/overlay_reprojection_zcwd_from_bag.cpp
)
target_link_libraries(overlay_reprojection_zcwd_from_bag
  ${PROJECT_NAME}_core
)
