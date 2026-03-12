#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('task_allocation'),
            'config',
            'stations.yaml'
        ]),
        description='Path to station configuration YAML file'
    )

    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='6',
        description='Number of robots in the fleet'
    )

    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value='map',
        description='Global reference frame ID'
    )

    robot_prefix_arg = DeclareLaunchArgument(
        'robot_prefix',
        default_value='robot',
        description='Namespace prefix for robots'
    )

    alpha_distance_arg = DeclareLaunchArgument(
        'alpha_distance',
        default_value='1.0',
        description='Weight coefficient for distance cost'
    )

    alpha_usage_arg = DeclareLaunchArgument(
        'alpha_usage',
        default_value='0.5',
        description='Weight coefficient for robot usage balancing'
    )

    alpha_battery_arg = DeclareLaunchArgument(
        'alpha_battery',
        default_value='2.0',
        description='Weight coefficient for battery penalty'
    )

    update_rate_arg = DeclareLaunchArgument(
        'update_rate_hz',
        default_value='2.0',
        description='Task allocation update rate in Hz'
    )

    task_allocation_node = Node(
        package='task_allocation',
        executable='task_allocation_node',
        name='task_allocation_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'num_robots': LaunchConfiguration('num_robots'),
                'global_frame': LaunchConfiguration('global_frame'),
                'robot_base_frame_prefix': LaunchConfiguration('robot_prefix'),
                'alpha_distance': LaunchConfiguration('alpha_distance'),
                'alpha_usage': LaunchConfiguration('alpha_usage'),
                'alpha_battery': LaunchConfiguration('alpha_battery'),
                'update_rate_hz': LaunchConfiguration('update_rate_hz'),
            }
        ]
    )

    return LaunchDescription([
        config_file_arg,
        num_robots_arg,
        global_frame_arg,
        robot_prefix_arg,
        alpha_distance_arg,
        alpha_usage_arg,
        alpha_battery_arg,
        update_rate_arg,
        task_allocation_node,
    ])
