"""Hover task."""

import numpy as np
from gym import spaces
import math
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to hover off the ground."""
#-------------------------------INITALIZE---------------------------------
    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,  0.0, -1.0, -1.0, -1.0,-1.0,-1.0,-1.0,-1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0,1.0,1.0,1.0]), dtype=np.float32)
        print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]),dtype=np.float32)
        print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0 # secs
        self.max_error_position = 8.0  # distance units
        self.target_position=np.array([0.0,0.0,10.0])#target hover position
        self.weight_position=0.65
        self.target_orientation=np.array([0.0,0.0,0.0,1.0])
        self.weight_orientation=0.32
        self.target_velocity=np.array([0.0, 0.0,0.0]) #target velocity
        self.weight_velocity=0.03
        
#---------------------------RESET-------------------------------------
    def reset(self):
        # Reset episode initial condition
        self.last_timestamp=None
        self.last_position=None
        p=self.target_position+np.random.normal(0.5,0.1,size=3)
        return Pose(
                position=Point(*p),  # slight random position
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )
#-------------------------------UPDATE-----------------------------------------
    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position=np.array([pose.position.x, pose.position.y, pose.position.z,])
        orientation=np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity=np.array([0.0,0.0,0.0])
        else:
            velocity=(position-self.last_position)/max(timestamp-self.last_timestamp,1e-03)
            
        state = np.concatenate([position,orientation,velocity])
        self.last_timestamp=timestamp
        self.last_position=position

        # Compute Error
        error_position=np.linalg.norm(self.target_position-state[0:3])
        error_orientation=np.linalg.norm(self.target_orientation-state[3:7])
        error_velocity=np.linalg.norm(self.target_velocity-state[7:10])**2
        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = (self.weight_position*error_position+self.weight_orientation*error_orientation+self.weight_velocity*error_velocity)  
        reward=-1*math.log(reward)# reward 
      
        if timestamp> self.max_duration/2:
            if error_position > self.max_error_position: #too far from target
                reward -= 3.0 #Yes 
                done=True
            elif timestamp > self.max_duration:
                reward += 5.0  #stayed close during time limit
                done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
