import os.path
import copy

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from SelfDriveEnv import Car, Track
from config import cfg


# ============================= #
#  Testing framework functions  #
# ============================= #


def run(model, env, render=True):
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

