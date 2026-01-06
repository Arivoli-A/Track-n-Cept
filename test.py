from IBVS import image_visual_servo
from MPC import MPC_track
from src.env_robot import Env, Robot

import numpy as np

def scenario1(function, velocities=None):
    if velocities is None:
        velocities = [-1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0,-9.0,-10.0]

    success_list = []
    for velocity in velocities:
        env = Env(GUI = True, puck_velocity=[0.0, velocity, 0.0])
        robot = Robot()
        
        success = function(env, robot, record = False)
        success_list.append(success)

    success_rate = np.mean(success_list)
    print("Scenario 1: Success rate: ", success_rate)
    return success_list, success_rate

def scenario2(function, velocities=None):
    if velocities is None:
        velocities = [1.0,2.0,3.0,4.0,5.0,6.0] # 7.0,8.0,9.0,10.0 - after 6.0 collision physics failing

    success_list = []
    for velocity in velocities:
        env = Env(GUI = True, puck_velocity=[velocity, -3.5, 0.0])
        robot = Robot()
        
        success = function(env, robot, record = False)
        success_list.append(success)

    success_rate = np.mean(success_list)
    print("Scenario 2: Success rate: ", success_rate)
    return success_list, success_rate

def tests(function, file_name):
    scenario1_success, scenario1_rate = scenario1(function)
    scenario2_success, scenario2_rate = scenario2(function)

    # Write results to log file
    with open(file_name + "_results.log", "w") as f:
        f.write("Scenario 1:\n")
        f.write(f"Success list: {scenario1_success}\n")
        f.write(f"Success rate: {scenario1_rate}\n\n")

        f.write("Scenario 2:\n")
        f.write(f"Success list: {scenario2_success}\n")
        f.write(f"Success rate: {scenario2_rate}\n")

    print(f"Results written to {file_name + "results.log"}")


if __name__ == "__main__":
    # tests(image_visual_servo, "IBVS")
    tests(MPC_track, "MPC")
