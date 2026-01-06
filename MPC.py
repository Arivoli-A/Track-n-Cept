import numpy as np
import casadi as ca
import pybullet as p
import pybullet_data
import cv2

import imageio

import src.utils as utils

DT = 1.0/240
N = 20
MAX_ITER = 2000
NQ = 7
z_desired = 0
IS_CONTACT_TOL = 1e-1

def predict_puck(x0, env, N = N, dt = DT):
    """
    3D puck rollout with elastic collisions.
    x0 = [x, y, z, vx, vy, vz]
    """
    x_min, x_max = -env.table_width/2, env.table_width/2
    y_min, y_max = -np.inf, env.table_length
    z_min, z_max = env.puck_height/2, np.inf

    x, y, z, vx, vy, vz = x0
    traj = [np.array([x, y, z, vx, vy, vz])]

    for k in range(N):

        # free motion
        nx = x + vx*dt
        ny = y + vy*dt
        nz = z + vz*dt
        nvx, nvy, nvz = vx, vy, vz

        # X collisions
        if nx <= x_min:
            nvx = -nvx
            nx = x_min + (x_min - nx)
        elif nx >= x_max:
            nvx = -nvx
            nx = x_max - (nx - x_max)

        # Y collisions
        if ny <= y_min:
            nvy = -nvy
            ny = y_min + (y_min - ny)
        elif ny >= y_max:
            nvy = -nvy
            ny = y_max - (ny - y_max)

        # Z collisions
        if nz <= z_min:
            nvz = -nvz
            nz = z_min + (z_min - nz)
        elif nz >= z_max:
            nvz = -nvz
            nz = z_max - (nz - z_max)

        traj.append(np.array([nx, ny, nz, nvx, nvy, nvz]))
        x, y, z, vx, vy, vz = nx, ny, nz, nvx, nvy, nvz   # Modificationn of velocity happens here

    return np.stack(traj, axis=0)



def pixel_to_camera(u, v, Z, f = utils.camera_focal_depth, cx = utils.camera_width / 2 , cy = utils.camera_height / 2):
    Xc = (u - cx) * Z / f
    Yc = (v - cy) * Z / f
    Zc = Z
    return np.array([Xc, Yc, Zc])

def camera_to_world(p_cam, cam_position, cam_orientation):
    # Transform 3D point from camera frame to world frame.
    
    return cam_position + cam_orientation @ p_cam

def estimate_velocity(pos, last_pos, last_vel, dt = DT, alpha=0.5):
    """
    Estimate filtered puck velocity.
    alpha: smoothing factor 0-1
    """

    if last_pos is None:
        vel_raw = np.zeros(3)
    else:
        vel_raw = (pos - last_pos) / dt

    vel = alpha * vel_raw + (1 - alpha) * last_vel

    return vel

def update_puck_state(u, v, Z, cam_position, cam_orientation, last_pos, last_vel):
    # Compute puck 3D position and velocity from raw camera observation.
    
    p_cam = pixel_to_camera(u, v, Z)

    pos_world = camera_to_world(p_cam, cam_position, cam_orientation)

    # velocity estimation
    vel_world = estimate_velocity(pos_world, last_pos, last_vel)

    return pos_world, vel_world


def puck_rollout(u, v, Z,
               cam_position, cam_orientation, 
               last_pos, last_vel,
               env,
               dt=DT,  N_pred=N):
    """
    One function that:
        - reads raw camera data (u, v, Z)
        - converts to world 3D
        - estimates velocity
        - predicts future motion
    """

    # update position + velocity
    pos, vel = update_puck_state(
        u, v, Z,
        cam_position, cam_orientation, last_pos, last_vel)

    # CasADi prediction
    x0 = np.hstack([pos, vel])
    traj = predict_puck(x0, env, N=N_pred, dt=dt)

    return pos, vel, traj

def puck_cam_frame(q, puck_world):
    """
    Transform single puck 3D world position into camera(EE) frame
    Inputs:
        q : (7,) or casadi vector
        puck_world : (3,)
    Returns:
        puck_cam : (3,)
    """
    T = utils.ee_pose(q)
    R = T[:3,:3]
    t = T[:3,3]

    return R.T @ (puck_world - t)


def build_MPC(N=N, nq=NQ):

    # ===== Parameters =====
    Q0 = ca.SX.sym("Q0", nq)
    puck_traj = ca.SX.sym("puck", 3, N)
    delay = 10 # Delay chosen based on time constant (3*tau - 5*tau) of the second order system caused by the PD control in robot
    # ===== Joint limits (CasADi constants) =====
    q_min  = ca.DM([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, 0])
    q_max  = ca.DM([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525, 0])
    qd_max = ca.DM([ 2.1750,  2.1750,  2.1750,  2.1750,  2.61,    2.61,   0])

    # ===== Decision variables =====
    q  = ca.SX.sym("q", nq, N+1)
    qd = ca.SX.sym("qd", nq, N)

    cost = 0
    g = []
    lbg = []
    ubg = []

    # ===== Initial condition (equality) =====
    g.append(q[:, 0] - Q0)
    lbg += [0]*nq
    ubg += [0]*nq

    for k in range(N):

        # EE pose   
        if k>delay:  # To include the dealy caused by PD control of robot
            T_ee = utils.ee_pose(q[:, k-delay])
        else:
            T_ee = utils.ee_pose(q[:, 0])
        ee_pos = T_ee[:3, 3]
        cam_z = T_ee[:3, 2]

        # Cost
        dist = puck_traj[:, k] - ee_pos
        eps = 1e-6
        align = (1 - ca.dot(dist, cam_z) / (ca.norm_2(dist)+eps))**2
        cost += 10*ca.dot(dist, dist) + 100*align  + 1e-3 * ca.dot(qd[:, k], qd[:, k])

        # ===== Dynamics (equality) =====
        g.append(q[:, k+1] - (q[:, k] + qd[:, k]*DT))
        lbg += [0]*nq
        ubg += [0]*nq

        # ===== Joint limits (inequality) =====
        g.append(q[:, k] - q_min)     # q >= q_min
        lbg += [0]*nq
        ubg += [ca.inf]*nq

        g.append(q_max - q[:, k])     # q <= q_max
        lbg += [0]*nq
        ubg += [ca.inf]*nq

        # ===== Velocity limits (inequality) =====
        g.append(qd_max - qd[:, k])   # qd <= qd_max
        lbg += [0]*nq
        ubg += [ca.inf]*nq

        g.append(qd[:, k] + qd_max)   # qd >= -qd_max
        lbg += [0]*nq
        ubg += [ca.inf]*nq

    # ===== Stack everything =====
    G = ca.vertcat(*g)

    X = ca.vertcat(
        ca.reshape(q, nq*(N+1), 1),
        ca.reshape(qd, nq*N, 1)
    )

    P = ca.vertcat(Q0, ca.reshape(puck_traj, 3*N, 1))

    prob = {"f": cost, "x": X, "g": G, "p": P}

    solver = ca.nlpsol(
        "solver", "ipopt", prob,
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
    )

    return solver, np.array(lbg), np.array(ubg)




def MPC(solver, lbg, ubg, Q0, puck_traj_world, X_prev=None):

    # Build parameter vector
    P = np.concatenate([
        Q0,
        puck_traj_world[:N].reshape(-1)
    ])

    if X_prev is None:
        sol = solver(p=P, lbg=lbg, ubg=ubg)
    else:
        sol = solver(x0=X_prev, p=P, lbg=lbg, ubg=ubg)

    Xopt = sol["x"].full().flatten()

    off = NQ*(N+1)
    qd0 = Xopt[off : off+NQ]

    return qd0, Xopt





def MPC_track(env, robot, record = True):

    frames = []
    last_pos = None
    last_vel = np.zeros(3)
    q = None

    solver, lbg, ubg = build_MPC()
    X_prev = None

    eye_in_hand_frames = []
    env_frames = []

    for ITER in range(MAX_ITER):
        p.stepSimulation()
        
        eePosition, eeOrientation = robot.get_ee_position()
        cameraOrientation = eeOrientation 
        cameraPosition = eePosition + np.array([0.01, 0.01, 0.01])
        # print('EE position (pybullet):', eePosition)
        rgb, depth, segment = utils.get_camera_img_float(cameraPosition, cameraOrientation)
        
        utils.draw_coordinate_frame(cameraPosition, cameraOrientation, 0.1)

        # Get object location from segmentation image
        
        object_loc = utils.get_puck_center_from_camera(env.puck_id, segmentation_image=segment) 

        if object_loc:
            
            u, v = object_loc

            z_error = z_desired - depth[v, u]

            puck_pos, puck_vel, puck_traj = puck_rollout(u, v, depth[v, u], cameraPosition, cameraOrientation, last_pos, last_vel, env)

            # print('Puck Velocity', puck_vel)
            # print('Puck trajectory', puck_traj)

            q0 = robot.get_current_joint_angles()

            # MPC control
            delta_q, X = MPC(solver,lbg,ubg,q0,puck_traj[:,:3],X_prev)

            X_prev = X
            nq = NQ
            
            q_opt = X[:nq*(N+1)].reshape((N+1, nq))  # Extract joint positions from Xopt

            
            puck_traj_cam = np.zeros((N+1, 3)) 

            for k in range(N+1):
                puck_traj_cam[k] = np.squeeze(puck_cam_frame(q_opt[k], puck_traj[k,:3]))  

            # print('Puck trajectory cam', puck_traj_cam)
            
            q = q0 + delta_q*DT

            # for q_trial in q_opt:
            #     print('EE', utils.ee_pose(q_trial)[:3,3])

            last_pos = puck_pos
            last_vel = puck_vel
            
        if q.any():
            # for i, idx in enumerate(robot._active_joint_indices):
            #     p.resetJointState(robot.robot_id, idx, q[i])
            robot.set_joint_position(q)
        
        # print(q)
        
        # show image
        if object_loc:
            u, v = object_loc

        if record:
            cv2.circle(rgb, (int(utils.camera_width/2), int(utils.camera_height/2)), 5, (0, 255, 0), -1)
            cv2.circle(rgb, (u,v), 5, (255, 0, 255), -1)
            cv2.imshow("depth", depth)
            cv2.imshow("rgb", rgb)
            eye_in_hand_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            env_frames.append(utils.record_camera())
            cv2.waitKey(1)

        #coordinate of the puck
        puck_position, puck_orientation = p.getBasePositionAndOrientation(env.puck_id)
        # print(puck_position)

        if abs(z_error) < IS_CONTACT_TOL and puck_position[1] >= env.finish_line:
            success=True
            break

        if puck_position[1] < env.finish_line:
            success=False
            break

    if record:
        imageio.mimsave('robot_servoing_MPC.gif', eye_in_hand_frames, fps=20)
        imageio.mimsave('robot_servoing_MPC_env.gif', env_frames, fps=20)
        print("Gif saved")

    #close the physics server
    cv2.destroyAllWindows()    
    p.disconnect() 

    return success


if __name__ == "__main__":
    from src.env_robot import Env, Robot
    env = Env() # puck_velocity=[0.0, -3.5, 0.0]
    robot = Robot()
    
    MPC_track(env, robot)