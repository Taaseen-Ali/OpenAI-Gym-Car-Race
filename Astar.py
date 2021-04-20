#!/usr/bin/env python

import heapq
import copy
import sys


class Env:    
    """Abstract class describing the interface used by the A* algorithm below.
    This allows for a generalized solution that can be plugged into a variety of
    problems as long as they implement these methods
    """

    def set_state(self):
        raise NotImplementedError("Class %s doesn't implement set_state" % (self.__class__.__name__))
    
    def get_state(self):
        raise NotImplementedError("Class %s doesn't implement get_state" % (self.__class__.__name__))
    
    def get_actions(self):
        raise NotImplementedError("Class %s doesn't implement get_action" % (self.__class__.__name__))

    def step(self, action):
        raise NotImplementedError("Class %s doesn't implement step" % (self.__class__.__name__))

    def heuristic(self):
        raise NotImplementedError("Class %s doesn't implement heuristic" % (self.__class__.__name__))

    def print_state(self, state):
        raise NotImplementedError("Class %s doesn't implement print_state" % (self.__class__.__name__))
    
    def done(self):
        raise NotImplementedError("Class %s doesn't implement done" % (self.__class__.__name__))
        

class Node:
    """Encapsulates a node to be used in an A* search tree
    """
    
    def __init__(self, parent, state, g, h, action=None):                
        """This class is used to wrap all of the information needed by A* and
        calculate the f value

        Args: 
            parent (Node): Reference to the parent of this Node
            state (*): Description of the current state of the problem
                represented by this node
            g (int): g cost to this node (path cost)
            h (int): h cost to this node (heuristic cost)
            action (*, optional): The action which led to this node being
                generated. Defaults to None for the root state. Should be equal to a
                member of the action space otherwise
        """

        self.parent = parent
        self.state = copy.deepcopy(state)
        self.hash = hash(str(self.state))
        self.action = action
        self.g = g
        self.h = h
        self.f = self.g + self.h

    def __lt__(self, other):
        return self.f < other.f    

    def __gt__(self, other):
        return self.f > other.f
    
    def __le__(self, other):
        return self.f <= other.f
    
    def __ge__(self, other):
        return self.f >= other.f


def run_astar(start_node, env):
    """Execute the A* algorithm using a valid environment and starting state

    Args:
        start_node (Node): Starting node to search from
        env (Env): Environment that describes the transition model to use
    """
    
    def traverse_back(node, path=[]):
        """Helper function to recursively get the path from the root to the node

        Args:
            node (Node): Current node 
            path (list, optional): Current path accumulated. Defaults to [].

        Returns:
            [Node]: List containing the path of the nodes in correct order
        """
        
        path.append(node)
        if node.action:
            return traverse_back(node.parent, path)
        else:
            path.reverse()
            return path
    
    # Initialize the tree with just the root node and an empty auxiliary hash
    # table
    
    frontier = [start_node]
    num_nodes = 1
    node_history = {}
    done = False
    
    # While the search state has not been found...
    
    while not done:
        
        # ...pop the node with the lowest f cost...
        
        parent = heapq.heappop(frontier)                                
        
        # and expand it only if it is not a repeated state with a worse path
        # cost. Save the f cost if is better than a previously explored path to
        # the same state        
        if node_history.get(parent.hash, float("inf")) > parent.f:                         
            node_history[parent.hash] = parent.f
        else:
            continue
        
        # Set the state of the transition model to the one currently being
        # observed 
        env.set_state(parent.state)
        
        # If it is an end state, return the needed info
        
        if env.done():
            return traverse_back(parent), num_nodes
        
        # Otherwise, get expand the node by getting all valid actions,
        # calculating the new g and h costs, and pushing them to the frontier
        
        actions = env.get_actions()
        for action in actions:            
            env.step(action)
            heapq.heappush(frontier, Node(parent, env.get_state(), 1 + parent.g, env.heuristic(), action))    
            num_nodes += 1
            env.set_state(parent.state)


def output_to_file(answer, file):
    """Utility function to output A* search results in the desired format

    Args:
        env (Env): Environment that was searched
        answer ((Node, integer)): Tuple containing a list of nodes representing
            the optimal path and number of nodes generated during the search
        file (string)): Output file name
    """
    
    # Set stdout to be the output file    

    with open(file, 'w') as file:
        sys.stdout = file
        
        # Extract the path, number of nodes, root and leaf nodes and optimal
        # path depth from the search results
        
        path, n = answer    
        root = path[0]
        leaf = path[-1]
        d = len(path) - 1    

        print("Depth of answer solution: ", d)
        print("Number of nodes generated: ", n)
        
        # Print the list of actions in order
        
        print("Solution")
        print("========")
        for node in path[1:]:
            print(node.action.value, end=" ")
        print()
        
        # Print the f costs in order
        
        print("F-costs")
        print("========")
        for node in path:
            print(node.f, end=" ")
    
    # Reset stdout
    sys.stdout = sys.__stdout__