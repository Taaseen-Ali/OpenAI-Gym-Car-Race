import os
import argparse

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.training_utils import TensorboardCallback, linear_schedule
from gym_car_race.config import cfg

Node:
    state: dict
    """
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
    """
    GCost
    FCost
    prev_node

env = Track()
car = Car()
actions = car.get_actions()
env.add_car(car)
obs = env.reset(new=args.ifreset)

def execute_astar(track):
    start = Node(env.get_state())
    explored = set()  # Set of nodes already explored, hashed for key
    solution = []
    start.setGCost(0)  # Path cost for start node is 0
    startNode.setFCost(heuristic + GCost)
    frontier = [start]  # Create frontier list, initialize with start node
    heapq.heapify(frontier)

    while(len(frontier) > 0 and len(solution) == 0):
        node = heapq.heappop(frontier)
        if goal == node:
            make solution
        else:
            explored.add(node)
            for child, action in generate_nodes(node, previous node):
                if child not in explored:
                    calculate GCost
                    calculate FCost
                    heapq.heappush(frontier, child)

def generate_nodes(env, node, actions):
    env.set_state(node.state)
    for action in actions:
        child = env.step(action).get_state()
        listOfChildNodes.append(child)

    return listOfChildNodes
