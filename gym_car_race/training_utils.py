import os.path
import copy

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.config import cfg


# ============================= #
#  Testing framework functions  #
# ============================= #


def run(model, env, render=True):
    """Runs trained model in env until done flag is raised

    Parameters:
        model: (stable_baselins3.BaseAlgorithm) 
            Trained model to be run (i.e. the model returned by .load)
        env: (SelfDriveEnv)
            Environment for the model to be run on. Car should be added
            before running this function
        render: (bool) 
            True to show the track and car as they run. False to only show
            text output.
    """
    
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs) 
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if render:
            env.render()

    print("finished with a reward of %d" % total_reward)
    return total_reward


def run_experiment(*args, timesteps=10000, render=True, trials=1, run_after_training=False):
    """Executes a suite of tests passed in as parameters

    This function facilitates running multiple training cycles. The caller would
    specify the various models to train/test by passing ina a variable number of
    arguments (*args) each being a function that takes in an integer
    representing the number of training timesteps as a parameter and returns a
    tuple of length 3 containing the model to be run, the environment to be run
    on, and a description of the test. It is recomended to use the testing api
    to help with this. (see "testing" function
    below)

    *args: (callable[[int], (stable_baselines3.BaseAlgorithm, SelfDriveEnv,
        str)]) Variable number of functions which each return a tuple containing
        the model, environment and a description 
    timesteps: (int) 
        Number of timesteps to train each model for.
    render: (bool)
        True to render each simulation after training, false for text only output
    trials: (int)
        The number of times to run each test if run_after_training is set to True
    run_after_training: (bool)
        True to run each test after training, false otherwise
    """
    to_run = []
    num_tests = len(args)
    print("Running %d models with %d training timesteps each for a total of %d timesteps" % (num_tests, timesteps, num_tests * timesteps))

    for env_test in args:
        to_run.append(env_test(timesteps))
        if run_after_training:
            model, env, _ = to_run[-1]
            run(model, env)

    for (model, env, desc) in to_run:
        print("Running model testing \"%s\"" % desc)        
        sum_rewards = 0
        for i in range(trials):
            print("\t - Trial #%d:" % (i+1), end=" ", flush=True)            
            reward = run(model, env, render=render)
            sum_rewards += reward
        print("\t Average reward over %d trials: %d\n" % (trials, sum_rewards/trials))


def testing(desc, config, save_as=None, in_dir=None):
    """Creates a test function from config

    This function returns another function which when called, trains a model
    according to the provided config. If the model has already been trained, it
    is loaded and returned instead.

    Parameters:
        desc: (str)
            Description of the test being defined
        config: (dict)
            Dictionary containing all of the necessary config for the SelfDriveEnv
        save_as: (str)
            Name to save/load the trained model as
        in_dir: (str)
            Directory to save/load the model in
    
    Returns: A function of the form int->(stable_baslines3.BaseAlgorithm, SelfDriveEnv, str)
    """
    def train(timesteps): 
        env = Track(config)
        car = Car(config)
        env.add_car(car)

        if save_as and in_dir and os.path.exists("%s/%s.zip" % (in_dir, save_as)):
            return PPO.load(in_dir + "/" + save_as), env, desc

        print("Testing %s" % desc)
        model = PPO('MlpPolicy', env, tensorboard_log=config["training"]["log_dir"], verbose=1, learning_rate=config["training"]["learning_rate"]) 
        model.learn(total_timesteps=timesteps, callback=TensorboardCallback(), tb_log_name=save_as) 
        
        if save_as and in_dir:
            model.save(in_dir + "/" + save_as)    
        return model, env, desc
    
    return train


def with_changes(changes):
    """Creates config from default and specified changes

    This function takes in any subset of configuration as defined in config.py
    and returns a new config dictionary with all of the specified changes
    applied to it

    Paramters:
        changes: (dict)    
            Dictionary specifying the changes to be made to the default configs
            found in config.py
    """
    
    updated = copy.deepcopy(cfg)
    for key in changes.keys():
        if key not in updated:
            raise KeyError("'%s' does not exist in config.py" % key)
        for to_change in changes[key].keys():                        
            if to_change not in updated[key]:
                raise KeyError("'%s.%s' does not exist config.py" % (key, to_change))
            updated[key][to_change] = changes[key][to_change]
    return updated


# ========================== #
#  Learning rate schedulers  #
# ========================== #


def constant_schedule(learning_rate): 
    """
    Constant learning rate schedule.

    learning_rate: (float)
        Value to use as learning rate for the duration of training
    
    returns: schedule that maintains a fixed value
    """
    
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        return: current learning rate
        """
        return learning_rate         
    return func


def linear_schedule(initial_value: float): 
    """
    Linear learning rate schedule.

    initial_value: (float) 
        Initial learning rate
    
    returns: schedule that computes current learning rate depending on remaining
        progress
    """
    
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        return current learning rate
        """

        return progress_remaining * initial_value         
    return func


# =============== #
# Logging Helpers #
# =============== #


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.count = 0
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        
        value = sum(self.training_env.get_attr("cars")[0][0].reward_history)
        self.logger.record('reward', value)
        return True

