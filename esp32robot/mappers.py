from esp32robot.utils import ConfigurationParameters


def map_config_params_to_dict(args: ConfigurationParameters):
    return {
        # environment parameters
        'env_type':args.environment_config.get('env_type', 'maze'),
        'stack_size':args.environment_config.get('stack_size', 4),
        'latency_steps':args.environment_config.get('latency_steps', 5),
        'rotation_penalty':args.environment_config.get('rotation_penalty', 0.1),
        'threshold_blocked_front':args.environment_config.get('thresholds', {}).get('blocked_front', 0),
        'threshold_corner':args.environment_config.get('thresholds', {}).get('corner', 0),
        'threshold_proximity_danger':args.environment_config.get('thresholds', {}).get('proximity_danger', 0),
        'threshold_idle_movement':args.environment_config.get('thresholds', {}).get('idle_movement', 0),
        'rewards':args.environment_config.get('rewards', None),
        
        # agent parameters
        'use_dqn':args.agent_config.get('use_dqn', True),
        'learning_rate':args.agent_config.get('learning_rate', 0.0005),
        'gamma':args.agent_config.get('gamma', 0.9),
        'batch_size':args.agent_config.get('batch_size', 32),
        'epsilon_start':args.agent_config.get('epsilon_start', 1.0),
        'min_epsilon':args.agent_config.get('min_epsilon', 0.05),
        'epsilon_decay':args.agent_config.get('epsilon_decay', 0.99),
        'memory_size':args.agent_config.get('memory_size', 10000),
        'target_update_freq':args.agent_config.get('target_update_freq', 100),
        
        # world model parameters
        'world_model_path':args.world_model_config.get('model_path', 'mini_world_model'),
        'world_model_n_categories':args.world_model_config.get('n_categories', 32),
        'world_model_hidden_size':args.world_model_config.get('hidden_size', 64),
        
        #training parameters
        'episodes':args.training_config.get('episodes', 50),
        'max_steps':args.training_config.get('max_steps', 30),
        'display_every':args.training_config.get('display_every', 5),
        'headless':args.training_config.get('headless', False),
        'model_name':args.training_config.get('model_name', 'q_table_pc'),
        'experiment_name':args.training_config.get('experiment_name', 'ESP32_Robot_Navigation_2D'),
        'output_dir':args.training_config.get('output_dir', ''),
    }