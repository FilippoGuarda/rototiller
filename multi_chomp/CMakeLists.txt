cmake_minimum_required(VERSION 3.8)
project(multi_chomp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# add dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)        
find_package(rosidl_default_generators REQUIRED) 
find_package(tf2_ros REQUIRED) 

# first we generate required interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MultiChompOptimize.action"
  DEPENDENCIES geometry_msgs nav_msgs
)

# create the executables ONCE INTERFACES ARE CREATED OR IT DOES NOT WORK
add_executable(multi_chomp_action_server
  src/multi_chomp.cpp
  src/multi_chomp_action_server.cpp 
  src/multi_chomp_action_server_main.cpp
)

# linking libraries
ament_target_dependencies(multi_chomp_action_server
  rclcpp
  rclcpp_action
  geometry_msgs
  visualization_msgs
  nav_msgs
  Eigen3
  OpenCV
  tf2_ros
)

target_include_directories(multi_chomp_action_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(multi_chomp_action_server  ${OpenCV_LIBRARIES})

# link 
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(multi_chomp_action_server "${cpp_typesupport_target}")

install(TARGETS multi_chomp_action_server
  DESTINATION lib/${PROJECT_NAME}
)

# python script install
install(PROGRAMS
  scripts/multi_chomp_coordinator.py
  DESTINATION lib/${PROJECT_NAME}
)

# launch file install
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# config file install
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
