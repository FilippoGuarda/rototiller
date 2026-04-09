#!/usr/bin/env python3

# Copyright 2026 Filippo Guarda
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_dir = FindPackageShare('multi_chomp')
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_dir, 'config', 'multi_chomp_params.yaml']),
        description='Path to config file'
    )

    # fleet path deconfliction
    coordinator_node = Node(
        package='multi_chomp',
        executable='multi_chomp_coordinator_original.py', 
        name='fleet_coordinator',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_count': 6} # Explicitly set robot count here if needed
        ]
    )

    # multi chomp server
    server_node = Node(
        package='multi_chomp',
        executable='multi_chomp_action_server',
        name='multi_chomp_server',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/global_costmap/costmap', '/robot1/global_costmap/costmap')
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        coordinator_node, 
        server_node        
    ])
