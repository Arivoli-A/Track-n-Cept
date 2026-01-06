import src.utils as utils
from src.env_robot import Env, Robot
import numpy as np
import pybullet as p
import time

env = Env(puck_velocity=[0.0, -3.5, 0.0])
robot = Robot()


q_check = [
    np.zeros(7),  # home / zero config

    np.array([0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.8]),  # nominal working pose

    np.array([0.3, -0.7, 0.2, -2.0, 0.1, 1.8, -0.5]),

    np.array([-0.4, -1.0, 0.5, -2.3, -0.2, 2.2, 1.2]),

    np.array([0.6, -0.3, -0.6, -1.8, 0.4, 1.5, -1.0]),

    np.array([0.000, -0.369, 0.369, -0.369, 0.369, 1.154, 0.369]) 
]


for q in q_check:
    np.set_printoptions(formatter={'float_kind':lambda x: f"{0.0 if abs(x)<1e-6 else x:.3f}"})
    
    print('EE position (FK):', utils.ee_pose(q)[:3,3])   
    
    # print('EE position (FK):')
    # utils.ee_pose(q)   
    
    robot.set_joint_position(q)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1/240)
    
    # print('EE position (pybullet):')
    # robot.get_ee_position_test()

    eePosition, eeOrientation = robot.get_ee_position()
    print('EE position (pybullet):', eePosition)
    print('Difference:', eePosition - utils.ee_pose(q)[:3,3])
    
    print('Joint angles',robot.get_current_joint_angles())

    print('\n')


