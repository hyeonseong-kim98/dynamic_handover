cmake_minimum_required(VERSION 3.8)
project(dyn_ho_ctl)

set(CMAKE_PREFIX_PATH "/opt/ros/humble;${CMAKE_PREFIX_PATH}")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# ROS 2 라이브러리 경로를 직접 지정
set(ROS2_LIB_PATH "/opt/ros/humble/lib")
set(ROS2_INCLUDE_PATH "/opt/ros/humble/include")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ur_rtde REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBMODBUS REQUIRED libmodbus)

# Include directories
include_directories(
  include
  ${ROS2_INCLUDE_PATH}
  ${EIGEN3_INCLUDE_DIR}
  ${LIBMODBUS_INCLUDE_DIRS}
)

link_directories(${ROS2_LIB_PATH})

# Create library for rg gripper
add_library(rg_gripper_client 
  src/rg_gripper_client.cpp
)

target_include_directories(rg_gripper_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(rg_gripper_client 
  ${LIBMODBUS_LIBRARIES}
)


# Create library for PD controller
add_library(pd_controller
  src/pd_controller.cpp
)

target_include_directories(pd_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(pd_controller
  Eigen3::Eigen
)

# Create library for dynamic handover control
add_library(dyn_ho_ctl
  src/dyn_ho_ctl.cpp
)

target_include_directories(dyn_ho_ctl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(dyn_ho_ctl
  pd_controller
  ur_rtde::rtde
  ${rclcpp_LIBRARIES}
  Eigen3::Eigen
  ${LIBMODBUS_LIBRARIES}
  rg_gripper_client
)

# Create executable
add_executable(dyn_ho_ctl_node src/main.cpp)
target_link_libraries(dyn_ho_ctl_node
  dyn_ho_ctl
  ${LIBMODBUS_LIBRARIES}
  rg_gripper_client
)

# Install
install(TARGETS
  pd_controller
  dyn_ho_ctl
  dyn_ho_ctl_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

ament_package()