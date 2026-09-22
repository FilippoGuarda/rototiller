import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, EmitEvent
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

SEED = 15
NUM_TASKS = 15

RUN_DURATION_SEC = 300.0  # benchmark runs terminate 5 minutes after launch

def generate_launch_description():
    task_allocation_dir = get_package_share_directory('task_allocation')
    graph_generator_dir = get_package_share_directory('graph_generator_node')
    multi_chomp_dir = get_package_share_directory('multi_chomp')
    
    # Import config file for task allocation
    stations_config = os.path.join(task_allocation_dir, 'config', 'stations_random.yaml')

    # All logs for this method go into their own folder
    log_dir = os.path.join(os.getcwd(), 'logs', 'rolling_chomp', 'r6', 'throwaway')
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f'task_allocation_log_ours_t{NUM_TASKS}_s{SEED}.csv')
    rolling_chomp_metrics_path = os.path.join(log_dir, f'rolling_chomp_metrics_ours_t{NUM_TASKS}_s{SEED}.csv')
    
    # launch graph generator and rolling chomp before running the task allocation stack
    graph_gen_launch = os.path.join(graph_generator_dir, 'launch', 'graph_generator.launch.py')
    # TODO: CHANGE OURS TO ORIGINAL WHEN TESTING AGAINST EXTENDED SPADES
    rolling_chomp_launch = os.path.join(multi_chomp_dir, 'launch', 'rolling_chomp.launch.py')
    
    launch_description = LaunchDescription()

    # ===== BENCHMARK TIMEOUT =====
    # Shuts down the entire stack (graph generator, rolling chomp, task allocation)
    # RUN_DURATION_SEC seconds after this launch starts.
    launch_description.add_action(TimerAction(
        period=RUN_DURATION_SEC,
        actions=[EmitEvent(event=Shutdown(
            reason=f'Benchmark finished: {RUN_DURATION_SEC:.0f}s timeout reached'))],
    ))

    # ===== GRAPH GENERATOR =====
    # Generates skeleton graph from occupancy grid
    graph_gen_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(graph_gen_launch)
    )
    launch_description.add_action(graph_gen_launch_include)

    # ===== rolling CHOMP (rolling) =====
    # rolling robot navigation with collision avoidance
    rolling_chomp_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rolling_chomp_launch),
        launch_arguments={
            'logfilepath': rolling_chomp_metrics_path,
            'runid': 'ours',
        }.items()
    )
    launch_description.add_action(rolling_chomp_launch_include)

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
                'run_id': 'ours',
            },
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
            'seed': SEED,
            'num_tasks': NUM_TASKS,
            'min_delay_s': 2.0,
            'max_delay_s': 8.0
        }],
        remappings=[
            ('skeleton_graph_json', '/skeleton_graph_json'),
            ('/tasks', '/tasks'),
        ],
    )
    launch_description.add_action(task_publisher_node)
    
    return launch_description