from gym_car_race.SelfDriveEnv import Car, Track, Utils
import Astar

class SearchableTrack(Astar.Env):
    def __init__(self, track):
        self.track = track
        self.car = self.track.cars[0]    
    
    def set_state(self, state):
        self.track.set_state(state)
    
    def get_state(self):
        return self.track.get_state()
    
    def get_actions(self):
        if self.car.crashed:
            return []
        forward = [(self.car.ACCELERATE, self.car.REST)] * 10
        left = [(self.car.ACCELERATE, self.car.ACCEL_LEFT)] * 10
        right = [(self.car.ACCELERATE, self.car.ACCEL_RIGHT)] * 10
        return [forward, left, right]

    def step(self, actions):        
        for move in actions:            
            self.track.step(move)
            self.track.render()
        
    def heuristic(self):
        return Utils.dist(self.track.finish_locs, self.get_state()["pos"])

    def print_state(self, state):
        pass
    
    def done(self):
        return self.get_state()["has_finished"]


# Initialize a track

track = Track()
car = Car()
track.add_car(car)
track.reset(new=False)

# Wrap it to be searched

to_search = SearchableTrack(track)

# Run A* and save the output to a file

start_node = Astar.Node(None, to_search.get_state(), 0, to_search.heuristic())
Astar.output_to_file(Astar.run_astar(start_node, to_search), "output.txt")