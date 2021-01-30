
from SelfDriveEnv import Car, Track
import config as cfg

env = Track(cfg.track['num_blocks_x'], cfg.track['num_blocks_y'], 
            cfg.track['block_width'], cfg.track['block_height'])
car = Car(cfg.car['position'][0], cfg.car['position'][1], 
          cfg.car['num_sensors'])
env.add_car(car)
env.reset(True)

model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=40000)


env = Track(cfg.track['num_blocks_x'], cfg.track['num_blocks_y'], 
            cfg.track['block_width'], cfg.track['block_height'])
car = Car(cfg.car['position'][0], cfg.car['position'][1], 
          cfg.car['num_sensors'])
env.add_car(car)
obs = env.reset()
done = False
total_reward = 0

print("Running simulation...")

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward    
    # This is optional
    env.render()

print("Car finished with a reward of", total_reward) 

""