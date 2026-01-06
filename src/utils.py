#import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import cv2
from scipy.spatial.transform import Rotation as Rot            #can use this to apply angular rotations to coordinate frames

import sympy as sp

import imageio

import casadi as ca

#camera (don't change these settings)
camera_width = 512                                             #image width
camera_height = 512                                            #image height
camera_fov = 120                                                #field of view of camera

camera_focal_depth = 0.5*camera_height/np.tan(0.5*np.pi/180*camera_fov) #focal depth in pixel space
                                                               

camera_aspect = camera_width/camera_height                     #aspect ratio
camera_near = 0.02                                             #near clipping plane in meters, do not set non-zero
camera_far = 100                                               #far clipping plane in meters


#control objectives (if you wish, you can play with these values for fun)
object_location_desired = np.array([camera_width/2,camera_height/2])  #center the object to middle of image
                                                                
K_p_x = 0.1                                                    #Proportional control gain for translation
K_p_Omega = 0.02                                               #Proportional control gain for rotation       




def pseudo_inverse(J):

    m, n = J.shape
    factor = 1e-2
    
    if m >= n:
        # (JTJ)-1 J
        JTJ = J.T @ J + factor**2*np.eye(J.shape[1])
        return np.linalg.inv(JTJ) @ J.T
    else:
        # JT(JJT)-1
        JJT = J @ J.T + factor**2*np.eye(J.shape[0])
        return J.T @ np.linalg.inv(JJT)


def draw_coordinate_frame(position, orientation, length, frameId = []):
    '''
    Draws a coordinate frame x,y,z with scaled lengths on the axes 
    in a position and orientation relative to the world coordinate frame
    pos: 3-element numpy array
    orientation: 3x3 numpy matrix
    length: length of the plotted x,y,z axes
    frameId: a unique ID for the frame. If this supplied, then it will erase the previous location of the frame
    
    returns the frameId
    '''
    if len(frameId)!=0:
        p.removeUserDebugItem(frameId[0])
        p.removeUserDebugItem(frameId[1])
        p.removeUserDebugItem(frameId[2])
    
    lineIdx=p.addUserDebugLine(position, position + np.dot(orientation, [length, 0, 0]), [1, 0, 0])  # x-axis in red
    lineIdy=p.addUserDebugLine(position, position + np.dot(orientation, [0, length, 0]), [0, 1, 0])  # y-axis in green
    lineIdz=p.addUserDebugLine(position, position + np.dot(orientation, [0, 0, length]), [0, 0, 1])  # z-axis in blue

    return lineIdx,lineIdy,lineIdz

def opengl_plot_world_to_pixelspace(pt_in_3D_to_project, viewMat, projMat, imgWidth, imgHeight):
    ''' Plots a x,y,z location in the world in an openCV image
    This is used for debugging, e.g. given a known location in the world, verify it appears in the camera
    when using p.getCameraImage(...). The output [u,v], when plot with opencv, should line up with object 
    in the image from p.getCameraImage(...)
    '''
    pt_in_3D_to_project = np.append(pt_in_3D_to_project,1)
    #print('Point in 3D to project:', pt_in_3D_to_project)

    pt_in_3D_in_camera_frame = viewMat @ pt_in_3D_to_project
    #print('Point in camera space: ', pt_in_3D_in_camera_frame)

    # Convert coordinates to get normalized device coordinates (before rescale)
    uvzw = projMat @ pt_in_3D_in_camera_frame
    #print('after projection: ', uvzw)

    # scale to get the normalized device coordinates
    uvzw_NDC = uvzw/uvzw[3]
    #print('after normalization: ', uvzw_NDC)

    #x,y specifies lower left corner of viewport rectangle, in pixels. initial value is (0,)
    u = ((uvzw_NDC[0] + 1) / 2.0) * imgWidth
    v = ((1-uvzw_NDC[1]) / 2.0) * imgHeight

    return [int(u),int(v)]

    
def get_camera_view_and_projection_opencv(cameraPos, camereaOrn):
    '''Gets the view and projection matrix for a camera at position (3) and orientation (3x3)'''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    _, _, rgbImg, depthImg, _ = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    #returns camera view and projection matrices in a form that fits openCV
    viewMat = np.array(__camera_view_matrix_opengl).reshape(4,4).T
    projMat = np.array(__camera_projection_matrix_opengl).reshape(4,4).T
    return viewMat, projMat

def get_camera_img_float(cameraPos, camereaOrn):
    ''' Gets the image and depth map from a camera at a position cameraPos (3) and cameraOrn (3x3) in space. '''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    width, height, rgbImg, nonlinDepthImg, segmentation = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Convert to numpy arrays
    rgbImg = np.array(rgbImg, dtype=np.uint8).reshape(height, width, 4)
    nonlinDepthImg = np.array(nonlinDepthImg, dtype=np.float32).reshape(height, width)
    segmentation = np.array(segmentation, dtype=np.float32).reshape(height, width)
    
    #adjust for clipping and nonlinear distance i.e., 1/d (0 is closest, i.e., near, 1 is furthest away, i.e., far
    depthImgLinearized =camera_far*camera_near/(camera_far+camera_near-(camera_far-camera_near)*nonlinDepthImg)

    #convert to numpy and a rgb-d image
    rgb_image = np.array(rgbImg[:,:,:3], dtype=np.uint8)
    depth_image = np.array(depthImgLinearized, dtype=np.float32)
    segmentation_image = np.array(segmentation, dtype=np.float32)

    return rgb_image, depth_image, segmentation_image


def get_puck_center_from_camera(puck_id, segmentation_image):
    """
    Captures a camera image from PyBullet, extracts segmentation mask,
    and returns center of puck in pixel coordinates.

    Returns:
        (u, v) pixel coordinates
        None   if puck not visible
    """

    H, W = segmentation_image.shape

    # Find all pixels belonging to the puck_id
    ys, xs = np.where(segmentation_image == puck_id)

    if len(xs) == 0:
        return None   # puck not visible

    # Average pixel location = center
    u = int(np.mean(xs))   # horizontal (column) - pixel value
    v = int(np.mean(ys))   # vertical   (row) - pixel value

    return (u, v)

def record_camera(camTargetPos = [-0.36, 0.72, -0.33], camDistance = 3, camOrientation = [44.4, -51, 0]):
    
    # Camera setup

    # CamOrientation - yaw, pitch, roll

    width, height = 640, 480
    camTargetPos = camTargetPos
    camDistance = camDistance
    yaw, pitch, roll = camOrientation


    # Recording parameters 
    duration = 10.0            # seconds
    fps = 30                   # frames per second in GIF
    sim_timestep = 1.0 / 240.0 # physics step
    steps_per_frame = int(1.0 / (fps * sim_timestep))  # e.g., 8 steps per frame
    total_frames = int(duration * fps)

    # Get camera image for this frame
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, 2)
    projMatrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 100)
    _, _, px, _, _ = p.getCameraImage(width, height, viewMatrix, projMatrix, renderer=p.ER_TINY_RENDERER)

    # Extract RGB channels only
    rgb = np.reshape(px, (height, width, 4))[:, :, :3]
    rgb = rgb.astype(np.uint8) 
    
    return rgb




def transl(x, y, z, is_casadi=False):
    T= [
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ]
    
    if is_casadi:
        T = ca.SX(T)  
    else:
        T = np.array(T)
    
    return T

def rot_x(a, is_casadi):
    if is_casadi:
        T = ca.SX([
            [1, 0, 0, 0],
            [0, ca.cos(a), -ca.sin(a), 0],
            [0, ca.sin(a),  ca.cos(a), 0],
            [0, 0, 0, 1]
        ])
    else:
        T = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a), -np.sin(a), 0],
            [0, np.sin(a),  np.cos(a), 0],
            [0, 0, 0, 1]
        ])
    return T

def rot_y(a, is_casadi):
    if is_casadi:
        c = ca.cos(a)
        s = ca.sin(a)
        T = ca.SX([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])
    else:
        c = np.cos(a)
        s = np.sin(a)
        T = np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])
    return T

def rot_z(a, is_casadi):
    if is_casadi:
        c = ca.cos(a)
        s = ca.sin(a)
        T = ca.SX([
            [ c, -s, 0, 0],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1]
        ])
    else:
        c = np.cos(a)
        s = np.sin(a)
        T = np.array([
            [ c, -s, 0, 0],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1]
        ])
    return T


def rpy_to_rot(rpy, is_casadi):
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]
    return rot_x(r, is_casadi) @ rot_y(p, is_casadi) @ rot_z(y, is_casadi)

def axis_rotation(axis, theta):
    """
    Returns a 4x4 homogeneous rotation matrix for rotation about an arbitrary axis. Using Rodrigues formula
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # Detect CasADi symbolic input
    is_casadi = isinstance(theta, ca.SX) or isinstance(theta, ca.MX)

    if is_casadi:
        # Convert axis components to SX
        x = ca.SX(axis[0])
        y = ca.SX(axis[1])
        z = ca.SX(axis[2])
        c = ca.cos(theta)
        s = ca.sin(theta)

        T = ca.vertcat(
            ca.horzcat(c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0),
            ca.horzcat(y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s, 0),
            ca.horzcat(z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c), 0),
            ca.horzcat(0, 0, 0, 1))

    else:
        x, y, z = axis
        c = np.cos(theta)
        s = np.sin(theta)
        T = np.array([
            [c + x*x*(1-c),     x*y*(1-c) - z*s,  x*z*(1-c) + y*s, 0],
            [y*x*(1-c) + z*s,   c + y*y*(1-c),    y*z*(1-c) - x*s, 0],
            [z*x*(1-c) - y*s,   z*y*(1-c) + x*s,  c + z*z*(1-c),   0],
            [0,                  0,                0,               1]
        ])

    return T




def ee_pose(q):
    """FK for Panda with camera offset, works with NumPy or CasADi"""
    camera_offset = 0.1
    urdf_params = [
        {'xyz':[0,0,0.333],      'rpy':[0,0,0],        'axis':[0,0,1]},
        {'xyz':[0,0,0],          'rpy':[-np.pi/2,0,0], 'axis':[0,0,1]},
        {'xyz':[0,-0.316,0],     'rpy':[np.pi/2,0,0],  'axis':[0,0,1]},
        {'xyz':[0.0825,0,0],     'rpy':[np.pi/2,0,0],  'axis':[0,0,1]},
        {'xyz':[-0.0825,0.384,0],'rpy':[-np.pi/2,0,0], 'axis':[0,0,1]},
        {'xyz':[0,0,0],          'rpy':[np.pi/2,0,0],  'axis':[0,0,1]},
        {'xyz':[0.088,-0.15,0],      'rpy':[np.pi/2,0,0],  'axis':[0,0,1]},
    ]
    
    is_casadi = isinstance(q, ca.SX) or isinstance(q, ca.MX)

    T = ca.SX.eye(4) if is_casadi else np.eye(4)
    
    for i in range(7):
        param = urdf_params[i]
        T_joint = transl(*param['xyz'], is_casadi) @ rpy_to_rot(param['rpy'], is_casadi) @ axis_rotation(param['axis'], q[i])
        T = T @ T_joint
        # print(T)
        # print('\n')
        
    # Camera offset along EE z-axis
    T[:3,3] = T[:3,3] + camera_offset * T[:3,2]
    return T
