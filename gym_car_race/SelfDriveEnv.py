#!/usr/bin/env python

import sys
import pygame
from math import *

import numpy as np
import gym
from gym import spaces

from gym_car_race.config import cfg


class Track(gym.Env): 
    """OpenAI gym environment simulating a car on a racetrack

    Most of the general configuration can be tweaked in config.py. Cars must be
    added to this track via add_car before training/running. Only one car can
    currently be added to the track but there are plans for multi agent support.
    In order to create a new track, call reset(new=True). Tracks are saved
    automatically and overide any previously saved track. To load a previously
    saved track, omit the "new" parameter when calling reset. 

    Attributes:
        action_space:       (gym.spaces.MultiDiscrete)
            Action space of 2 discrete values representing acceleration and turn
            direction. See "actions" in config.py for more details.
        observation_space:  (gym.spaces.Box(0.0, inf, (n), float32))
            Observation space of n values corresponding to the distance from each of
            the n sensor in the car
        track:              ([[TrackBorder...]])
            Two dimensional array representing the track.
        cars:               ([Car...])
            List of cars that have been added to the track. Currently only the first
            one is used
        start_locs:         ([int...])
            List of starting line TrackBorders
        finish_locs:        ([int...])
            List of finish line TrackBorders
    """

    def __init__(self, config=cfg):
        """
        Parameters:
            num_blocks_x: (int)
                Number of horizontal tiles in the track
            num_blocks_y: (int)
                Number of vertical tiles in the track
            block_width: (int)
                Width of a single tile
            block_height: (int)
                height of a single tile
        """    
        
        super(Track, self).__init__()
        pygame.init()
        pygame.display.set_icon(pygame.image.load("gym_car_race/images/logo.png"))
        pygame.display.set_caption("Gym Car Race -- NYU SelfDrive", "Gym Car Race")        
        
        self.action_space = spaces.MultiDiscrete([3,3])
        """ self.action_space = DiscreteActions.get_action_space() """
        self.observation_space = spaces.Box(np.zeros((config["car"]["num_sensors"] + 2)), \
            np.full((config["car"]["num_sensors"] + 2), inf))        
        
        self._num_blocks_x, self._num_blocks_y = config["track"]['num_blocks_x'], config["track"]['num_blocks_y']
        self._block_width, self._block_height = config["track"]['block_width'], config["track"]['block_height']
        
        self._initialized = False
        
        self._clock = pygame.time.Clock()
        self._screen_width, self._screen_height = self._block_width * self._num_blocks_x, \
            self._block_height * self._num_blocks_y
        self._screen = None
        
        self.track_file = config["track"]["track_file"]
        self.track = [[TrackBorder(x*self._block_width, y*self._block_height, self._block_width,
            self._block_height, (x,y)) for x in range(-1,self._num_blocks_x+1)
            ] for y in range(-1, self._num_blocks_y+1)]
        
        self.cars = []
        self.start_locs = []    #start coordinates
        self.finish_locs = []   #finish coordinates
        
    def _calc_avg_pos(self, start_finish):
        """Calculates the average of either start_locs or finish_locs
        coordinates

        Parameters:
            start_finish: ("start" | "finish") Specifies which array you
            want to average
        """
        
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

    def open_window(self):
        self._screen = pygame.display.set_mode((self._screen_width, 
            self._screen_height))
        
    def close_window(self):
        pygame.display.quit()
        self._screen = None
    
    def load_track(self):
        with open(self.track_file) as f:
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
        with open(self.track_file, "a") as f:
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
        """Event handler for pygame events.

        Extend this if you wish to add some functionality that requires
        keyboard/mouse input while running the simulation.
        """
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close_window()

    def colliding_with(self, x, y):    
        for row in self.track:
            for border in row:
                if border.rect.collidepoint(x,y) and border.active:                    
                    return True, border.start_finish, border.index
                    
        return False, None, None
    
    def current_tile(self, car):
        x, y = car.get_car_center()
        for row in self.track:
            for border in row:
                if border.rect.collidepoint(x, y):
                    return border.index     

    def add_car(self, car):
        self.cars.append(car)
        car.set_track(self)

    def render(self):
        if not self._screen and self._initialized:
            self.open_window()
        self._screen.fill((30,30,30))
        self._clock.tick(60)
        for j in range(self._num_blocks_x+2):
            for k in range(self._num_blocks_y+2):
                tile = self.track[k][j]
                tile.render(self._screen)
        for car in self.cars:
            car.render(self._screen)        
        pygame.display.flip()
        self.handle_events()
    
    def reset(self, new=False):
        """OpenAI gym interface method for reseting the environment.
        
        Parameters:
            new: (bool) (optional)
                True to create new track, False to use a saved track
        
        Returns: () Observation from car sensors
        """

        if new:
            self.open_window()
            while self._screen:
                self.render()
            for row in self.track:
                for border in row:
                    border.mutable = False
            self.save_track()
        else:
            self.load_track()
        
        self.cars[0].reset()
        self.start_locs = self._calc_avg_pos("start")
        self.finish_locs = self._calc_avg_pos("finish")
        self.cars[0].set_pos(self.start_locs)
        self._initialized = True            
        return self.cars[0]._get_observation()
    
    def step(self, action):
        """OpenAI gym interface method to advance the simulation by one step

        Parmeters:
            action: (gym.spaces.MultiDiscrete([3 3])) 
                Next action to be applied to the car for this time step. The first
                member of action corresponds to forward/backward acceleration and the
                second corresponds to turn direction. Each member is a discrete
                value in the range [0, 2]
        """
        
        # TODO: Extend this to support multiple cars
        """ 
        if isinstance(action, np.ndarray):
            action = action[0]
        car = self.cars[0] """
        """ obs, reward, done, _ = car.step(DiscreteActions.get_controls_from_action(action)) """
        car = self.cars[0]
        obs, reward, done, _ = car.step(action)
        return obs, reward, done, _


class TrackBorder: 
    """Class representing a single block on the simulation

    A TrackBorder can have 4 main states: Active, inactive, start and finish. An
    active TrackBorder respresents a wall. Start and finish TrackBorders
    represent starting and finishing lines respectively. Inactive TrackBorders
    are not rendered and do not exist from the perspective of the car

    Attributes:
        dimenstions:    ((int, int, int, int))
            The x, y, width and height of the TrackBorder in pixels    
        rect:  (pygame.rect)
            Pygame rect representing this class
        start_color:    (int, int , int)
            Color of starting line TrackBorders
        finish_color:   (int, int, int)
            Color of finish line TrackBorders
        default_color:  (int, int, int)
            Color of regular active TrackBorders        
        border_color:   (int, int, int)
            Color of the outline on a TrackBorders        
        active: (bool)
            Represents whether or not the TrackBorder should be used
            rendering/collisions
        start_finish:   (string | None)
            Denotes whether the track is used in a starting line ("start"),
            finish line ("finish") or neither (None)
        index:          ([int, int])
            A list containing the row, col position of this TrackBorder in an
            array of TrackBorders

        mutable:        (bool)
            False to lock a TrackBorder from changes (like mouse clicks, for example)
    """
    
    def __init__(self, x, y, width, height, index):
        
        self.dimensions = (x, y, width, height)
        self.rect = pygame.Rect(*self.dimensions)
        
        self.start_color = cfg["track"]["start_line_color"]
        self.finish_color = cfg["track"]["finish_line_color"]
        self.default_color = cfg["track"]["default_color"]
        self.color = self.default_color
        self.border_color = cfg["track"]["border_color"]
        
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


class Car:
    """Class representing a car that can be added to the track environment
    
    Most of the general configuration is set via config.py, but you can also
    make some general tweaks here if necessary.

    Attributes:
        track:          (Track)
            Reference to the track that the car is active on
        width:          (int)
            Width of car bounding box
        height:         (int)
            Height of car bounding box
        image:          (pygame.image)
            Image representing the car
        angle:          (int)
            Direction of the car. 0 is straight up
        pos:            ([int, int])
            Pixel coords of the topleft box corner of the car
        sensors:        ([[int, int, int, int]...])
            Array of n sensors. Each sensor is a list of 4 numbers 
            representing the x positions representing the x coord, y coord, 
            angle from car tip, anddistance from car tip respectively
        traveled:       ([(int, int)...])
            List of (x, y) tuples. Each element corresponds to a TrackBorder 
            that has been reached by the car
        crashed:        (bool)
            True if the car has crashed, false otherwise
        has_finished:   (bool)
            True if car has finished the race, false otherwise
        done:           (bool)
            True if either the car has finished or crashed, false otherwise
        reward_history: ([int...])
            List of reward acheived per time step
    """

    def __init__(self, config=cfg):
        """
        Parameters:
            x: (int)
                Initial topleft horizontal pixel position of the car
            y: (int)
                Initial topleft vertical pixel position of the car
            num_sensors: (int)
        """
        self.config = config
        self.track = None
        self.image = pygame.image.load(config["car"]['image'])
        self.sensor_color = config["car"]["sensor_color"]
        self.width = config["car"]['width']
        self.height = config["car"]['height']
        self.image = pygame.transform.scale(self.image, (self.width, self.height))        
        self.angle = config["car"]['angle']
        self.pos = config["car"]["position"]
        
        self.crashed = False 
        self.has_finished = False
        self.done = False
        
        self.num_sensors = config["car"]["num_sensors"]
        self.sensors = [[0, 0,  i*(180.0/self.num_sensors), 0] for i in 
            range(self.num_sensors)]
        self._center_sensors()
        self.speed = config["car"]['speed']
        self.rotation = config["car"]['rotation']

        self.acceleration = config["car"]['acceleration']
        self.max_speed = config["car"]['max_speed']

        self.turn_rate = config["car"]['turn_rate']
        self.max_turn_rate = config["car"]['max_turn_rate']
        
        self.REST = config["action"]['rest']
        self.DECELERATE = config["action"]['decelerate']
        self.ACCELERATE = config["action"]['accelerate']
        self.ACCEL_LEFT = config["action"]['accel_left']
        self.ACCEL_RIGHT = config["action"]['accel_right']
    
        self.NEW_TILE_REWARD = config["reward"]['new_tile_reward']
        self.SAME_TILE_REWARD = config["reward"]['same_tile_reward']
        self.CRASH_REWARD = config["reward"]['crash_reward']
        self.traveled = []
        self._calc_reward = config["reward"]["function"](self) if config["reward"]["function"] else self._default_step_reward
        self.reward_history = []

    @staticmethod
    def reward_function(f):
        """Decorator for creating custom reward functions

        This wrapper simply allows the Car class to pass in an instance of
        itself into the reward function without calling it. Doing so allows us
        to dynamically set the reward function and call it without passing in a
        reference of this car with each call. The function being wrapped must
        take only one parameter, a reference to an iniitalized car, and return
        an integer reward. Rewards for the last run are saved within an
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
                collided, start_finish, _ = self.track.colliding_with(sensor[0], sensor[1])
                if collided and not start_finish:
                    break                
                sensor[0] += 1*cos(-radians(self.angle + sensor[2]))
                sensor[1] += 1*sin(-radians(self.angle + sensor[2]))
            sensor[3] = Utils.dist(self.get_car_tip(), (sensor[0], sensor[1]))
        return [sensor[3] for sensor in self.sensors]
    
    def _get_observation(self):
        return np.array(self._update_sensors() + [self.speed, radians(self.angle)])
        """ return np.array(self._update_sensors())     """

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

    def has_collided(self):
        x, y = self.get_car_tip()
        return self.track.colliding_with(x, y)
    
    def move(self):
        if not self.done:
            self.angle += self.rotation
            self.pos = self._move_forward(self.speed)
            curr_tile = self.track.current_tile(self)
            
            reward = self._calc_reward()
            if curr_tile not in self.traveled:
                self.traveled.append(curr_tile)        
            return reward

        else: return self._calc_reward()
    
    def step(self, action):
        collided, tile_type, _ = self.has_collided()        
    
        self.has_finished = collided and tile_type == "finish"
        self.crashed = collided and not tile_type
        self.done = self.has_finished or self.crashed
        
        accel, rot = action

        if accel == self.ACCELERATE and self.speed < self.max_speed:
            self.speed += self.acceleration
        elif accel == self.DECELERATE and self.speed > self.acceleration:
            self.speed -= self.acceleration
        if rot == self.ACCEL_LEFT and self.rotation < self.max_turn_rate:            
            self.rotation += self.turn_rate
        elif rot == self.ACCEL_RIGHT and self.rotation > -self.max_turn_rate:
            self.rotation -= self.turn_rate
        
        reward = self.move()
        observations = self._get_observation()
        return observations, reward, self.done, {}

    def reset(self):
        self.pos = self.config["car"]['position']
        self.angle = self.config["car"]['angle']
        self._center_sensors()
        self.crashed = False 
        self.has_finished = False
        self.done = False
        self.reward_history = []

    def render(self, screen):
        rotated_image, new_rect = Utils.rotate_image(self.image, self.pos, self.angle)
        screen.blit(rotated_image, new_rect.topleft)
        for sensor in self.sensors:
            pygame.draw.circle(screen, self.sensor_color, (sensor[0], sensor[1]), 5)

    
class Utils:
    """Collection of commonly used utility methods."""
    
    @staticmethod
    def dist(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        return ((x0-x1)**2 + (y0-y1)**2)**.5

    @staticmethod
    def rotate(coord, angle, center):
        angle = -radians(angle+90)
        x0, y0 = center        
        d = Utils.dist(center, coord)
        return (d*cos(angle) + x0, d*sin(angle) + y0)

    @staticmethod
    def rotate_image(image, topleft, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center = image.get_rect(
            topleft = topleft).center)
        return rotated_image, new_rect
