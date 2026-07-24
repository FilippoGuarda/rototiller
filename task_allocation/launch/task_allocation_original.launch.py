import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    task_allocation_dir = get_package_share_directory('task_allocation')
    graph_generator_dir = get_package_share_directory('graph_generator_node')
    multi_chomp_dir = get_package_share_directory('multi_chomp')
    
    # Import config file for task allocation
    stations_config = os.path.join(task_allocation_dir, 'config', 'stations.yaml')
    # Set file address for logs, TODO: CHANGE OURS TO ORIGINAL WHEN TESTING AGAINST EXTENDED SPADES
    log_file_path = os.path.join(os.getcwd(), 'task_allocation_log_original_rand30.csv')
    multi_chomp_metrics_path = os.path.join(os.getcwd(), 'multi_chomp_metrics_original_rand30.csv')
    
    # launch graph generator and multi chomp before running the task allocation stack
    graph_gen_launch = os.path.join(graph_generator_dir, 'launch', 'graph_generator.launch.py')
    # TODO: CHANGE OURS TO ORIGINAL WHEN TESTING AGAINST EXTENDED SPADES
    multi_chomp_original_launch = os.path.join(multi_chomp_dir, 'launch', 'multi_chomp_original.launch.py')
    
    launch_description = LaunchDescription()
    
    # ===== GRAPH GENERATOR =====
    # Generates skeleton graph from occupancy grid
    graph_gen_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(graph_gen_launch)
    )
    launch_description.add_action(graph_gen_launch_include)
    # ===== MULTI CHOMP =====
    # Multi robot navigation with collision avoidance
    multi_chomp_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(multi_chomp_original_launch),
        launch_arguments={
            'logfilepath': multi_chomp_metrics_path,
            'runid': 'original',
        }.items()
    )
    launch_description.add_action(multi_chomp_launch_include)

    task_allocation_node = Node(
        package="task_allocation",
        executable="task_allocation_node.py",
        name="task_allocation_node",
        namespace="/",
        output="screen",
        parameters=[
            stations_config,
            # TODO: CHANGE RUN_ID TO ORIGINAL WHEN TESTING AGAINST EXTENDED SPADES
            {
                'log_file_path': log_file_path,
                'run_id': 'original',
            }
        ],
        remappings=[
            ("/skeleton_graph/graph_markers", "/skeleton_graph/graph_markers"),
            ("/tasks", "/tasks"),
        ],
    )
    launch_description.add_action(task_allocation_node)

    # random task publisher node
    task_publisher_node = Node(
        package="task_allocation",
        executable="task_publisher_node.py", 
        name="task_publisher_node",
        output="screen",
        parameters=[{
            'seed': 42,
            'num_tasks': 15,
            'min_delay_s': 2.0,
            'max_delay_s': 8.0
        }], 
        remappings=[
            ('skeleton_graph_json', '/skeleton_graph_json'),
            ('/tasks', '/tasks'),
        ]
    )
    launch_description.add_action(task_publisher_node)
    
    return launch_description
