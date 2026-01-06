import pybullet as p
import pybullet_data
import time
import numpy as np

import os

class Env:

    def __init__(self, GUI = True, puck_velocity = [3.0, -1.0, 0.0]):

        # Table parameters
        self.table_length = 4.0
        self.table_width = 1.0
        self.table_thickness = 0.05
        self.wall_height = 0.3
        self.wall_thickness = 0.05

        # Table transform
        self.table_position = (0, self.table_length/2, 0)

        # 90 rotation around Z (long axis along Y)
        self.table_orientation = (0, 0, 0.70710678, 0.70710678)

        # Puck parameters
        self.puck_radius = 0.06
        self.puck_height = 0.02
        self.puck_mass = 0.17
        self.puck_base_pos = [0.0, self.table_length-self.wall_thickness*2, self.puck_height/2 + 0.01]

        # PYBULLET INIT
        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)

 
        # CREATE WORLD OBJECTS
        self.create_long_table_with_edges()
        self.puck_id = self.create_puck()

        # Launch puck 
        p.resetBaseVelocity(self.puck_id, linearVelocity=puck_velocity)

        self.finish_line = 0.3 
        line_color = [1, 0, 0] # Red color [R, G, B]
        line_width = 3.0 # Optional width

        # Define the start and end points for a line spanning the x-axis
        lineFrom = [-1.0, self.finish_line, 0.1]
        lineTo = [1.0, self.finish_line, 0.1]

        # Add the debug line to the PyBullet environment
        # This will return a unique ID (line_id) which can be used to remove or update the line later.
        line_id = p.addUserDebugLine(lineFrom, lineTo, lineColorRGB=line_color, lineWidth=line_width)

        # You can also add a label:
        p.addUserDebugText("", [1.0, self.finish_line, 0.1], textColorRGB=line_color)


        # Optional engine tuning
        p.setPhysicsEngineParameter(contactSlop=0.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
 
    def create_long_table_with_edges(self):

        px, py, pz = self.table_position
        q = self.table_orientation

        
        L = self.table_length
        W = self.table_width
        T = self.table_thickness
        H = self.wall_height
        WT = self.wall_thickness

        table_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L/2, W/2, T/2]
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L/2, W/2, T/2],
            rgbaColor=[0.85, 0.85, 0.85, 1]
        )

        table_id = p.createMultiBody(
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[px, py, pz - T/2],
            baseOrientation=q
        )

        bodies = [table_id]

        # END WALLS (Y edges)
        end_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L/2, WT/2, H/2]
        )
        end_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L/2, WT/2, H/2],
            rgbaColor=[0.1, 0.1, 0.1, 1]
        )

        # +Y and -Y edges
        for y_sign in [-1, 1]:
            wy = py 
            wz = pz + H/2
            wx = px + y_sign * (W/2)

            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=end_col,
                baseVisualShapeIndex=end_vis,
                basePosition=[wx, wy, wz],
                baseOrientation=q
            )
            p.changeDynamics(wall_id, -1, lateralFriction=0.0, restitution=1.0)
            bodies.append(wall_id)

        # SIDE WALLS (X edges)
        side_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[WT/2, W/2, H/2]
        )
        side_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[WT/2, W/2, H/2],
            rgbaColor=[0.1, 0.1, 0.1, 1]
        )

        for x_sign in [1]:
            wx = px 
            wy = py + x_sign * (L/2)
            wz = pz + H/2

            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=side_col,
                baseVisualShapeIndex=side_vis,
                basePosition=[wx, wy, wz],
                baseOrientation=q
            )
            p.changeDynamics(wall_id, -1, lateralFriction=0.0, restitution=1.0)
            bodies.append(wall_id)

        return bodies

    def create_puck(self):
        puck_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.puck_radius,
            height=self.puck_height
        )
        puck_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.puck_radius,
            length=self.puck_height,
            rgbaColor=[1, 0, 0, 1]
        )

        puck_id = p.createMultiBody(
            baseMass=self.puck_mass,
            baseCollisionShapeIndex=puck_col,
            baseVisualShapeIndex=puck_vis,
            basePosition=self.puck_base_pos
        )
        
        p.changeDynamics(
                puck_id, -1,
                lateralFriction=0.01,
                spinningFriction=0.01,
                rollingFriction=0.01,
                restitution=1.0,
                linearDamping=0.0,
                angularDamping=0.0
            )

        return puck_id



    

# Robot with Camera Class
class Robot:
    def __init__(self, robot_id = None, initialJointPos = [0,-np.pi/4,np.pi/4,-np.pi/4,np.pi/4,np.pi/2,np.pi/4,0,0,0,0,0]):

        self.robot_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        
        self.eeFrameId = []
        self.camera_offset = 0.1 #offset camera in z direction to avoid grippers
        # Get the joint info
        self._numLinkJoints = p.getNumJoints(self.robot_id) #includes passive joint
        jointInfo = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        
        # Get joint locations (some joints are passive)
        self._active_joint_indices = []
        for i in range(self._numLinkJoints):
            if jointInfo[i][2]==p.JOINT_REVOLUTE:
                self._active_joint_indices.append(jointInfo[i][0])
        self.numActiveJoints = len(self._active_joint_indices) #exact number of active joints

        #reset joints
        for i in range(self._numLinkJoints):
            p.resetJointState(self.robot_id,i,initialJointPos[i])

        # while True:
        #     p.stepSimulation()
        #     time.sleep(1/240)


    def get_ee_position(self):
        '''
        Function to return the end-effector of the link. This is the very tip of the robot at the end of the jaws.
        '''
        endEffectorIndex = self.numActiveJoints
        endEffectorState = p.getLinkState(self.robot_id, endEffectorIndex)
        endEffectorPos = np.array(endEffectorState[0])
        endEffectorOrn = np.array(p.getMatrixFromQuaternion(endEffectorState[1])).reshape(3,3)
        
        #add an offset to get past the forceps
        endEffectorPos += self.camera_offset*endEffectorOrn[:,2]
        return endEffectorPos, endEffectorOrn

    def get_ee_position_test(self):
        '''
        Function to check the positions of each link
        '''
        endEffectorIndex = self._active_joint_indices
        for Index in endEffectorIndex:
            endEffectorState = p.getLinkState(self.robot_id, Index)
            print('endEffectorPos (','Index',')', np.array(endEffectorState[0]))
            print('endEffectorOrn (','Index',')',np.array(p.getMatrixFromQuaternion(endEffectorState[1])).reshape(3,3))
            
    def get_current_joint_angles(self):
        # Get the current joint angles
        joint_angles = np.zeros(self.numActiveJoints)
        for i in range(self.numActiveJoints):
            joint_state = p.getJointState(self.robot_id, self._active_joint_indices[i])
            joint_angles[i] = joint_state[0]
        return joint_angles
    
    def get_jacobian_at_current_position(self):
        #Returns the Robot Jacobian of the last active link
        mpos, mvel, mtorq = self.get_active_joint_states()   
        zero_vec = [0.0]*len(mpos)
        linearJacobian, angularJacobian = p.calculateJacobian(self.robot_id, 
                                                              self.numActiveJoints,
                                                              [0,0,self.camera_offset],
                                                              mpos, 
                                                              zero_vec,
                                                              zero_vec)
        #only return the active joint's jacobians
        Jacobian = np.vstack((linearJacobian,angularJacobian))
        return Jacobian[:,:self.numActiveJoints]
    
    def set_joint_position(self, desireJointPositions, kp=1.0, kv=0.3):
        '''Set  the joint angle positions of the robot'''
        zero_vec = [0.0] * self._numLinkJoints
        allJointPositionObjectives = [0.0]*self._numLinkJoints
        for i in range(desireJointPositions.shape[0]):
            idx = self._active_joint_indices[i]
            allJointPositionObjectives[idx] = desireJointPositions[i]

        p.setJointMotorControlArray(self.robot_id,
                                    range(self._numLinkJoints),
                                    p.POSITION_CONTROL,
                                    targetPositions=allJointPositionObjectives,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * self._numLinkJoints,
                                    velocityGains=[kv] * self._numLinkJoints)

    def get_active_joint_states(self):
        '''Get the states (position, velocity, and torques) of the active joints of the robot
        '''
        joint_states = p.getJointStates(self.robot_id, range(self._numLinkJoints))
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques



if __name__ == "__main__":
    env = Env()
    robot = Robot()
    
