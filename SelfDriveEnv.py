#!/usr/bin/env python

import sys
import pygame
from math import *
import numpy as np
import gym
from gym import spaces

import config as cfg

class Utils:
    """Collection of commonly used utility methods."""
    
    @staticmethod
    def dist(x0, y0, x1, y1):
        return ((x0-x1)**2 + (y0-y1)**2)**.5

    @staticmethod
    def rotate(coord, angle, center):
        angle = -radians(angle+90)
        x0, y0 = center
        x1, y1 = coord    
        d = Utils.dist(x0, y0, x1, y1)
        return (d*cos(angle) + x0, d*sin(angle) + y0)

    @staticmethod
    def rotate_image(image, topleft, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center = image.get_rect(
            topleft = topleft).center)
        return rotated_image, new_rect

class Car:
    """Class representing a car that can be added to the track environment

    ...
    
    Most of the general configuration is set via config.py, but you can also
    make some general tweaks here if necessary.

    Attributes:
        track:  (Track)
            Reference to the track that the car is active on
        width:  (int)
            Width of car bounding box
        height: (int)
            Height of car bounding box
        image:  (pygame.image)
            Image representing the car
        angle:  (int)
            Direction of the car. 0 is straight up
        pos:    ([int, int])
            Pixel coords of the topleft box corner of the car
        sensors:([[int, int, int, int]...])
            Array of n sensors. Each sensor is a list of 4 numbers 
            representing the x positions representing the x coord, y coord, 
            angle from car tip, anddistance from car tip respectively
        crashed:(bool)
            True if the car has crashed, false otherwise

    """

    def __init__(self, x, y, num_sensors, reward_func=None):
        self.track = None
        self.image = pygame.image.load(cfg.car['image'])
        self.width = cfg.car['width']
        self.height = cfg.car['height']
        self.image = pygame.transform.scale(self.image, (self.width, self.height))        
        self.angle = cfg.car['angle']
        self.pos = [x, y]        
        
        self.crashed = False 
        self.has_finished = False
        self.done = False
        
        self.sensors = [[0, 0,  i*(180.0/num_sensors), 0] for i in 
            range(num_sensors + 1)]
        self._center_sensors()
        self.speed = cfg.car['speed']
        self.rotation = cfg.car['rotation']

        self.acceleration = cfg.car['acceleration']
        self.max_speed = cfg.car['max_speed']

        self.turn_rate = cfg.car['turn_rate']
        self.max_turn_rate = cfg.car['max_turn_rate']
        
        self.REST = cfg.action['rest']
        self.DECELERATE = cfg.action['decelerate']
        self.ACCELERATE = cfg.action['accelerate']
        self.ACCEL_LEFT = cfg.action['accel_left']
        self.ACCEL_RIGHT = cfg.action['accel_right']
    
        self.NEW_TILE_REWARD = cfg.reward['new_tile_reward']
        self.SAME_TILE_REWARD = cfg.reward['same_tile_reward']
        self.CRASH_REWARD = cfg.reward['crash_reward']
        self.traveled = []
        self.calc_reward = reward_func(self) if reward_func else self._default_step_reward
        self.reward_history = []
    
    @staticmethod
    def reward_function(f):
        """Decorator for creating custom reward functions

        This decorator simply allows the Car class to pass in an instance of
        itself into the reward function without calling it. Doing so allows us
        to dynamically set the reward function and call it without passing in a
        reference of this car with each call. The function being wrapped must
        take only one parameter, a reference to an iniitalized car, and return
        an integer reward. Rewards for the last run are saved within and
        instance of the bound class
        """
        
        def bind_to(car):
            def wrapped():
                reward = f(car)
                car.reward_history.append(reward)
                return reward
            return wrapped
        return bind_to
    
    def _default_step_reward(self):        
        """Default reward function
        
        This function is used when no custom function is supplied. It can be
        tweaked using the various values found under "rewards" in config.py
        """
        
        reward = 0
        if self.crashed:
            reward = self.CRASH_REWARD
        else:
            curr_tile = self.current_tile()
            if curr_tile not in self.traveled:
                reward = self.NEW_TILE_REWARD        
            else:
                reward = self.SAME_TILE_REWARD
        self.reward_history.append(reward)
        return reward

    def _center_sensors(self):
        x, y = self.pos
        center = self.get_car_tip()
        for sensor in self.sensors:
            sensor[0], sensor[1] = center

    def _update_sensors(self):
        self._center_sensors()
        offset = 90
        for sensor in self.sensors:
            while True:
                if self.track.colliding_with(sensor[0], sensor[1]):
                    break                
                sensor[0] += 1*cos(-radians(self.angle + sensor[2]))
                sensor[1] += 1*sin(-radians(self.angle + sensor[2]))
            x0, y0 = self.get_car_tip()
            x1, y1, = sensor[0], sensor[1]
            sensor[3] = Utils.dist(x0, y0, x1, y1)
        return np.array([sensor[3] for sensor in self.sensors])
    
    def _move_forward(self, dist):
        offset = 90
        x, y = self.pos
        rad_angle = -radians(self.angle + offset)
        return (x + dist * cos(rad_angle), y + dist * sin(rad_angle))
        
    def set_pos(self, coords):
        self.pos = coords
    
    def set_track(self, track):
        self.track = track
    
    def get_car_tip(self):
        x, y = self.pos
        return Utils.rotate((x+self.width//2, y), self.angle, (x+self.width//2, y+self.height//2))

    def get_car_center(self):
        x, y = self.pos
        return x+25, y+25
    
    def current_tile(self):
        return self.track.current_tile(self)

    def has_crashed(self):
        x, y = self.get_car_tip()
        return self.track.colliding_with(x, y)
    
    def move(self):
        if not self.done:
            self.angle += self.rotation
            self.pos = self._move_forward(self.speed)
            curr_tile = self.track.current_tile(self)
            
            reward = self.calc_reward()
            if curr_tile not in self.traveled:
                self.traveled.append(curr_tile)        
            return reward

        else: return self.calc_reward()
    
    def step(self, action):
        pos = self.has_crashed()        
        
        if pos:
            row, col = pos
            self.has_finished = self.track.track[row][col].start_finish == "finish"        
            self.crashed = not self.has_finished
            self.done = True
        
        accel, rot = action

        if accel == self.ACCELERATE and self.speed < self.max_speed:
            self.speed += self.acceleration
        elif accel == self.DECELERATE and self.speed > 0:
            self.speed -= self.acceleration
        if rot == self.ACCEL_LEFT and self.rotation < self.max_turn_rate:            
            self.rotation += self.turn_rate
        elif rot == self.ACCEL_RIGHT and self.rotation > -self.max_turn_rate:
            self.rotation -= self.turn_rate
        
        reward = self.move()
        observations = self._update_sensors()
        return observations, reward, self.done, {}

    def reset(self):
        self.pos = cfg.car['position']
        self.angle = cfg.car['angle']
        self._center_sensors()

    def render(self, screen):
        color = (255, 255, 0)
        rotated_image, new_rect = Utils.rotate_image(self.image, self.pos, self.angle)
        screen.blit(rotated_image, new_rect.topleft)
        for sensor in self.sensors:
            pygame.draw.circle(screen, (255, 0, 0), (sensor[0], sensor[1]), 5)

    
class TrackBorder: 
    def __init__(self, x, y, width, height, index):
        self.dimensions = (x, y, width, height)
        self.rect = pygame.Rect(self.dimensions[0], self.dimensions[1],
            self.dimensions[2], self.dimensions[3])
        
        self.start_color = cfg.track["start_line_color"]
        self.finish_color = cfg.track["finish_line_color"]
        self.default_color = cfg.track["default_color"]
        self.color = self.default_color
        self.border_color = (255,255,255)
        
        self.active = True
        self.start_finish = None
        self.index = index
        self.mutable = True

    def check_state(self):
        mouse_pos = pygame.mouse.get_pos()
        pressed_state = pygame.mouse.get_pressed()
        keys = pygame.key.get_pressed()
        x, y, w, h = self.dimensions
        if x+w > mouse_pos[0] > x and y + h > mouse_pos[1] > y and self.mutable:
            if pressed_state[0]: 
                self.active = False
                self.start_finish = None                
            elif pressed_state[2]:
                self.active = True
                self.start_finish = None
            elif keys[pygame.K_s]:
                self.start_finish = "start"
            elif keys[pygame.K_f]:
                self.start_finish = "finish"
        if self.start_finish == "start":
            self.color = self.start_color
        elif self.start_finish == "finish":
            self.color = self.finish_color
        else: self.color = self.default_color
    
    def render(self, screen):
        self.check_state()
        if self.active:        
            pygame.draw.rect(screen, self.color, self.rect)    
            pygame.draw.rect(screen, self.border_color, self.rect, 1)


class Track(gym.Env): 
    def __init__(self, num_blocks_x, num_blocks_y, block_width, block_height):      
        super(Track, self).__init__()
        pygame.init()
        self.num_blocks_x, self.num_blocks_y = num_blocks_x, num_blocks_y
        self.action_space = spaces.MultiDiscrete([3,3])
        self.observation_space = spaces.Box(np.zeros((11)), \
            np.full((11), inf))
        self.initialized = False
        
        self.clock = pygame.time.Clock()
        self.screen_width, self.screen_height = block_width * num_blocks_x, \
            block_height * num_blocks_y
        self.screen = None
        
        self.track = [[TrackBorder(x*block_width, y*block_height, block_width,
            block_height, (x,y)) for x in range(-1,num_blocks_x+1)
            ] for y in range(-1, num_blocks_y+1)]
        self.cars = []
        self.start_locs = []    #start coordinates
        self.finish_locs = []   #finish coordinates
        
    def open_window(self):
        self.screen = pygame.display.set_mode((self.screen_width, 
            self.screen_height))
        
    def close_window(self):
        pygame.display.quit()
        self.screen = None
    
    def load_track(self):
        with open("track.csv") as f:
                content = f.readlines()
        content = [x.strip().split() for x in content] 
        for row in range(len(content)):
            for col in range(len(content[row])):
                self.track[row][col].mutable = False
                if content[row][col] == '0':
                    self.track[row][col].active = False
                else:
                    self.track[row][col].active = True
                    if content[row][col] == 's':
                        self.track[row][col].start_finish = "start"
                    elif content[row][col] == 'f':
                        self.track[row][col].start_finish = "finish"
    
    def save_track(self):
        with open("track.csv", "a") as f:
            f.seek(0)
            f.truncate()
            for row in self.track:
                for border in row:
                    if border.active and not border.start_finish:
                        f.write("1 ")
                    elif border.active and border.start_finish:
                        if border.start_finish == "start":
                            f.write("s ")
                        elif border.start_finish == "finish":
                            f.write("f ")
                    else: f.write("0 ")
                f.write("\n")
            
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close_window()

    def reset(self, new=False):
        if new:
            self.open_window()
            while self.screen:
                self.render()
            for row in self.track:
                for border in row:
                    border.mutable = False
            self.save_track()
        else:
            self.load_track()
        self.cars[0].reset()
        self.start_locs = self.calc_avg_pos("start")
        self.finish_locs = self.calc_avg_pos("finish")
        self.cars[0].set_pos(self.start_locs)
        self.initialized = True            
        return self.cars[0]._update_sensors()

    def colliding_with(self, x, y):    
        for row in self.track:
            for border in row:
                if border.rect.collidepoint(x,y) and border.active and \
                    not border.start_finish == "start":
                    return border.index
        return False
    
    def current_tile(self, car):
        x, y = car.get_car_tip()
        for row in self.track:
            for border in row:
                if border.rect.collidepoint(x, y):
                    return border.index     

    def render(self):
        if not self.screen and self.initialized:
            self.open_window()
        self.screen.fill((30,30,30))
        self.clock.tick(60)
        for j in range(self.num_blocks_x+2):
            for k in range(self.num_blocks_y+2):
                tile = self.track[k][j]
                tile.render(self.screen)
        for car in self.cars:
            car.render(self.screen)        
        pygame.display.flip()
        self.handle_events()
    
    def calc_avg_pos(self, start_finish):
        """calculates the average of the coordinates"""
        coords = []
        for row in self.track:
            for box in row:
                if box.start_finish == start_finish:
                    coords.append(box.dimensions[:2])
        if len(coords) > 1:
            x, y = 0, 0
            for coord in coords:
                x += coord[0]
                y += coord[1]
            x /= len(coords)
            y /= len(coords)
            coords = [x,y]
        else: coords = list(coords)
        return coords

    def add_car(self, car):
        self.cars.append(car)
        car.set_track(self)

    def step(self, action):
        """ for car in self.cars: """
        car = self.cars[0]
        obs, reward, done, _ = car.step(action)
        return obs, reward, done, _

