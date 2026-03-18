import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    """
    Launch description for enhanced task allocation system.
    
    Dynamically locates package directories using ament_index_python to handle 
    the nested src/extended_spades/... workspace structure.
    """
    
    # === 1. Resolve Package Directories ===
    task_allocation_dir = get_package_share_directory('task_allocation')
    graph_generator_dir = get_package_share_directory('graph_generator_node')
    multi_chomp_dir = get_package_share_directory('multi_chomp')
    
    # Import config file for task allocation
    stations_config = os.path.join(task_allocation_dir, 'config', 'stations.yaml')
    
    # launch graph generator and multi chomp before running the task allocation stack
    graph_gen_launch = os.path.join(graph_generator_dir, 'launch', 'graph_generator.launch.py')
    multi_chomp_launch = os.path.join(multi_chomp_dir, 'launch', 'multi_chomp.launch.py')
    
    launch_description = LaunchDescription()
    
    # ===== GRAPH GENERATOR =====
    # Generates skeleton graph from occupancy grid
    graph_gen_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(graph_gen_launch)
    )
    # launch_description.add_action(graph_gen_launch_include)
    # ===== MULTI CHOMP =====
    # Multi robot navigation with collision avoidance
    multi_chomp_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(multi_chomp_launch)
    )
    # launch_description.add_action(multi_chomp_launch_include)
    
    # ===== TASK ALLOCATION NODE =====
    # Main allocation coordinator with tropical optimization
    task_allocation_node = Node(
        package="task_allocation",
        executable="task_allocation_node.py",
        name="task_allocation_node",
        namespace="/",
        output="screen",
        parameters=[
            stations_config,
            # {
            #     "num_robots": 6,
            #     "global_frame": "/map",
            #     "robot_base_frame_prefix": "robot",
            #     "robot_base_frame_suffix": "/base_link",
            #     "node_match_threshold_m": 0.75,
            #     "k_nearest_graph_nodes": 3,
            #     "station_connection_distance": 2.0,
            #     "update_rate_hz": 2.0,
            #     "use_sim_time": True,
            #     "alpha_distance": 1.0,
            #     "alpha_usage": 0.5,
            #     "alpha_battery": 2.0,
            # },
        ],
        remappings=[
            ("/skeleton_graph/graph_markers", "/skeleton_graph/graph_markers"),
            ("/tasks", "/tasks"),
        ],
    )
    launch_description.add_action(task_allocation_node)

    # ===== TASK PUBLISHER NODE =====
    task_publisher_node = Node(
        package="task_allocation",
        executable="task_publisher_node.py", 
        name="task_publisher_node",
        output="screen",
        parameters=[{
            "task_id": "demo_sequence_001",
            "stations": ["stationa1", "stationc2"],
            "priority": 1.0,
            "publish_delay_s": 2.0, 
        }],
        remappings=[
            ("/tasks", "/tasks"),
        ],
    )
    launch_description.add_action(task_publisher_node)
    
    return launch_description
