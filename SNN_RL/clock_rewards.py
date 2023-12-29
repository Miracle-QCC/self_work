import numpy as np
import pickle
from cassie.trajectory.aslip_trajectory import get_ref_aslip_ext_state, get_ref_aslip_unaltered_state, get_ref_aslip_global_state
from sympy import symbols, Eq, solve,oo
from scipy.spatial.transform import Rotation
import math
def distance_to_circle(a, b, r, m, n):
    # 计算点到圆心的距离
    distance_to_center = math.sqrt((m - a)**2 + (n - b)**2)

    # 计算点到圆的最近距离
    closest_distance = distance_to_center - r

    return closest_distance
def central_angle(x0, y0, x1, y1, x2, y2):
    # 计算向量 (x1, y1) 到 (x0, y0) 的方向角
    angle1 = math.atan2(y1 - y0, x1 - x0)
    
    # 计算向量 (x2, y2) 到 (x0, y0) 的方向角
    angle2 = math.atan2(y2 - y0, x2 - x0)
    
    # 计算圆心角，注意要确保结果在 [0, 2*pi] 范围内
    angle = (angle2 - angle1) % (2 * math.pi)
    # angle = angle * 180 / 3.1415926
    return angle
def find_intersection_points(a, b, r, m, n):
    # 处理直线平行于 x 轴的情况
    if m == 0:
        delta_y = math.sqrt(r**2 - (a - m)**2)
        intersection_points = [(a, b + delta_y), (a, b - delta_y)]
    else:
        # 直线的斜率和截距
        k = (n - b) / (m - a)
        c = -k * a + b

        # 解二次方程得到交点的x坐标
        A = 1 + k**2
        B = 2 * (k * (c - b) - a)
        C = a**2 + (c - b)**2 - r**2

        discriminant = B**2 - 4 * A * C

        if discriminant >= 0:
            x1 = (-B + math.sqrt(discriminant)) / (2 * A)
            x2 = (-B - math.sqrt(discriminant)) / (2 * A)

            # 对应的y坐标
            y1 = k * x1 + c
            y2 = k * x2 + c

            # 选择x值较大的点
            intersection_points = (x1, y1) if x1 > x2 else (x2, y2)
        else:
            intersection_points = None

    return intersection_points
cnt = 0
def clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 250
    desired_max_foot_vel = 2.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0
    pelvis_distance_cost = 0
    # com orient error
    if qpos[0] <= 1.75:
        com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)
    else:
        a = 1.75
        b = 3
        r = 3
        intersection_point = find_intersection_points(a, b, r, qpos[0], qpos[1])
        if intersection_point is not None:
            angle = central_angle(a, b, 1.75, 0, intersection_point[0], intersection_point[1])
            # 欧拉角表示 (roll, pitch, yaw) in 弧度
            euler_angles = (0, 0, angle)
            # 创建 Rotation 对象并获取四元数
            rotation = Rotation.from_euler('xyz', euler_angles)
            quaternion = rotation.as_quat()
            neutral_pelvis_orient = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            com_orient_error +=10 * (1 - np.inner(neutral_pelvis_orient, qpos[3:7]) ** 2)
        pelvis_distance_cost = distance_to_circle(a, b, r, qpos[0], qpos[1])



    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - self.speed)
    
    # pelvis motion penalty : straight_diff, height deadzone, pelvis acceleration penalty
    straight_diff = np.abs(qpos[1])  # straight difference penalty
    if straight_diff < 0.05:
        straight_diff = 0

    #############################################
    height_diff = np.abs(qpos[2] - 0.9)  # height deadzone is range from 0.05 to 0.2 meters depending on speed
    # height = 0
    # if self.cassie_state.terrain.height > -0.11:
    #     height = self.cassie_state.terrain.height + 0.1187#爬楼梯奖励
    #############################################- cassie_state.terrain.height
    deadzone_size = 0.05 + 0.05 * self.speed
    if height_diff < deadzone_size:
        height_diff = 0
    pelvis_acc = 0.25 * (np.abs(self.cassie_state.pelvis.rotationalVelocity[:]).sum() + np.abs(self.cassie_state.pelvis.translationalAcceleration[:]).sum())
    pelvis_motion = straight_diff + height_diff + pelvis_acc

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)

    # scaled force/vel reward
    # left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    # left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    # right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    # right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    left_frc_score = np.tan(np.pi/4 * left_frc_clock * normed_left_frc)
    left_vel_score = np.tan(np.pi/4 * left_vel_clock * normed_left_vel)
    right_frc_score = np.tan(np.pi/4 * right_frc_clock * normed_right_frc)
    right_vel_score = np.tan(np.pi/4 * right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    # hip roll velocity penalty
    hip_roll_penalty = np.abs(qvel[6]) + np.abs(qvel[13])

    # torque cost
    torque = np.asarray(self.cassie_state.motor.torque[:])
    torque_penalty = 0.25 * (sum(np.abs(self.prev_torque - torque)) / len(torque))
    
    # action cost
    action_penalty = 5 * sum(np.abs(self.prev_action - action)) / len(action)
    
    # reward = 0.200 * foot_frc_score + \
    #          0.200 * foot_vel_score + \
    #          0.200 * np.exp(-(com_orient_error + foot_orient_error)) + \
    #          0.150 * np.exp(-pelvis_motion) + \
    #          0.150 * np.exp(-com_vel_error) + \
    #          0.050 * np.exp(-hip_roll_penalty) + \
    #          0.025 * np.exp(-torque_penalty) + \
    #          0.025 * np.exp(-action_penalty) + \
    #          0.2 * np.exp(-self.pelvis_distance_cost) 
    # if qpos[0] <= 1.75:
    #     reward += 0.5 * (qpos[0] - 1.75)

    ##直楼梯奖励函数
    # reward = 0.200 * foot_frc_score + \
    #         0.200 * foot_vel_score + \
    #         0.200 * np.exp(-(com_orient_error + foot_orient_error)) + \
    #         0.150 * np.exp(-pelvis_motion) + \
    #         0.050 * np.exp(-com_vel_error) + \
    #         0.050 * np.exp(-hip_roll_penalty) + \
    #         0.025 * np.exp(-torque_penalty) + \
    #         0.025 * np.exp(-action_penalty) + \
    #         0.15   * qpos[0]


    ##旋转楼梯奖励函数
    reward = 0.200 * foot_frc_score + \
            0.200 * foot_vel_score + \
            0.200 * np.exp(-foot_orient_error) + \
            0.200 * np.exp(-com_orient_error) + \
            0.150 * np.exp(-pelvis_motion) + \
            0.050 * np.exp(-com_vel_error) + \
            0.050 * np.exp(-hip_roll_penalty) + \
            0.025 * np.exp(-torque_penalty) + \
            0.025 * np.exp(-action_penalty) + \
            0.050 * np.exp(-pelvis_distance_cost)
    #self.debug
     
    if False:
        # print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        # print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))\
        global cnt
        cnt = cnt + 1
        if cnt >= 10000 and height != 0:
            cnt = 0
            print(
                f"reward:  \t{reward:.2f} / 1.000\n"
                f"foot_frc:\t{0.200 * foot_frc_score:.2f} / +-0.200\n"
                f"foot_vel:\t{0.200 * foot_vel_score:.2f} / +-0.200\n"
                f"orient:  \t{0.200 * np.exp(-(com_orient_error + foot_orient_error)):.2f} / 0.200\n"
                f"pelvis:  \t{0.150 * np.exp(-pelvis_motion):.2f} / 0.150\n"
                f"com_vel: \t{0.150 * np.exp(-com_vel_error):.2f} / 0.150\n"
                f"hip_roll:\t{0.050 * np.exp(-hip_roll_penalty):.2f} / 0.050\n"
                f"torque:  \t{0.025 * np.exp(-torque_penalty):.2f} / 0.025\n"
                f"action:  \t{0.025 * np.exp(-action_penalty):.2f} / 0.025\n"
                f"height:  \t{0.8    * height:.2f} / 1"
            )

            print("actual speed: {}\tcommanded speed: {}\n\n".format(qvel[0], self.speed))
    return reward

"""
Designed to speed up early parts of training, prevent bobbing in place local minima.
Changes to clock_reward:
- no pelvis acc
- much less weighting / less strict on torque, action, hip roll penalty, foot orient, com orient
- much more weighting on force/vel clock, com vel. Higher range on normalization of foot force/vel
"""
def early_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 350
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 1 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 1 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(self.speed - com_vel)
    
    # pelvis motion penalty : straight_diff, height deadzone, pelvis acceleration penalty
    straight_diff = np.abs(qpos[1])  # straight difference penalty
    if straight_diff < 0.05:
        straight_diff = 0
    height_diff = np.abs(qpos[2] - 0.9)  # height deadzone is range from 0.05 to 0.2 meters depending on speed
    deadzone_size = 0.05 + 0.05 * self.speed
    if height_diff < deadzone_size:
        height_diff = 0
    # pelvis_acc = 0.25 * (np.abs(self.cassie_state.pelvis.rotationalVelocity[:]).sum() + np.abs(self.cassie_state.pelvis.translationalAcceleration[:]).sum())
    pelvis_motion = straight_diff + height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)

    # scaled force/vel reward
    left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    # left_frc_score = np.tan(np.pi/4 * left_frc_clock * normed_left_frc)
    # left_vel_score = np.tan(np.pi/4 * left_vel_clock * normed_left_vel)
    # right_frc_score = np.tan(np.pi/4 * right_frc_clock * normed_right_frc)
    # right_vel_score = np.tan(np.pi/4 * right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    # hip roll velocity penalty
    hip_roll_penalty = np.abs(qvel[6]) + np.abs(qvel[13])

    # torque cost
    torque = np.asarray(self.cassie_state.motor.torque[:])
    torque_penalty = 0.25 * (sum(np.abs(self.prev_torque - torque)) / len(torque))
    
    # action cost
    action_penalty = 5 * sum(np.abs(self.prev_action - action)) / len(action)

    reward = 0.250 * foot_frc_score + \
             0.350 * foot_vel_score + \
             0.200 * np.exp(-com_vel_error) + \
             0.100 * np.exp(-(com_orient_error + foot_orient_error)) + \
             0.100 * np.exp(-pelvis_motion)
            #  0.050 * np.exp(-hip_roll_penalty) + \
            #  0.025 * np.exp(-torque_penalty) + \
            #  0.025 * np.exp(-action_penalty)

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))\
        print(
            f"reward:  \t{reward:.2f} / 1.000\n"
            f"foot_frc:\t{0.250 * foot_frc_score:.2f} / +-0.250\n"
            f"foot_vel:\t{0.250 * foot_vel_score:.2f} / +-0.250\n"
            f"com_vel: \t{0.200 * np.exp(-com_vel_error):.2f} / 0.200\n"
            f"orient:  \t{0.150 * np.exp(-(com_orient_error + foot_orient_error)):.2f} / 0.150\n"
            f"pelvis:  \t{0.150 * np.exp(-pelvis_motion):.2f} / 0.150\n"
            # f"hip_roll:\t{0.050 * np.exp(-hip_roll_penalty):.2f} / 0.050\n"
            # f"torque:  \t{0.025 * np.exp(-torque_penalty):.2f} / 0.025\n"
            # f"action:  \t{0.025 * np.exp(-action_penalty):.2f} / 0.025"
        )

        print("actual speed: {}\tcommanded speed: {}\n\n".format(qvel[0], self.speed))
    print("actual speed: {}\tcommanded speed: {}\n\n".format(qvel[0], self.speed))
    return reward

def no_speed_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 250
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - self.speed)
    
    # pelvis motion penalty : straight_diff, height deadzone, pelvis acceleration penalty
    straight_diff = np.abs(qpos[1])  # straight difference penalty
    if straight_diff < 0.05:
        straight_diff = 0
    height_diff = np.abs(qpos[2] - 0.9)  # height deadzone is range from 0.05 to 0.2 meters depending on speed
    deadzone_size = 0.05 + 0.05 * self.speed
    if height_diff < deadzone_size:
        height_diff = 0
    pelvis_acc = 0.25 * (np.abs(self.cassie_state.pelvis.rotationalVelocity[:]).sum() + np.abs(self.cassie_state.pelvis.translationalAcceleration[:]).sum())
    pelvis_motion = straight_diff + height_diff + pelvis_acc

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)

    # scaled force/vel reward
    # left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    # left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    # right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    # right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    left_frc_score = np.tan(np.pi/4 * left_frc_clock * normed_left_frc)
    left_vel_score = np.tan(np.pi/4 * left_vel_clock * normed_left_vel)
    right_frc_score = np.tan(np.pi/4 * right_frc_clock * normed_right_frc)
    right_vel_score = np.tan(np.pi/4 * right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    # hip roll velocity penalty
    hip_roll_penalty = np.abs(qvel[6]) + np.abs(qvel[13])

    # torque cost
    torque = np.asarray(self.cassie_state.motor.torque[:])
    torque_penalty = 0.25 * (sum(np.abs(self.prev_torque - torque)) / len(torque))
    
    # action cost
    action_penalty = 5 * sum(np.abs(self.prev_action - action)) / len(action)

    reward = 0.250 * foot_frc_score + \
             0.250 * foot_vel_score + \
             0.225 * np.exp(-(com_orient_error + foot_orient_error)) + \
             0.175 * np.exp(-pelvis_motion) + \
             0.050 * np.exp(-hip_roll_penalty) + \
             0.025 * np.exp(-torque_penalty) + \
             0.025 * np.exp(-action_penalty)

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))\
        print(
            f"reward:  \t{reward:.2f} / 1.000\n"
            f"foot_frc:\t{0.200 * foot_frc_score:.2f} / +-0.200\n"
            f"foot_vel:\t{0.200 * foot_vel_score:.2f} / +-0.200\n"
            f"orient:  \t{0.200 * np.exp(-(com_orient_error + foot_orient_error)):.2f} / 0.200\n"
            f"pelvis:  \t{0.150 * np.exp(-pelvis_motion):.2f} / 0.150\n"
            f"hip_roll:\t{0.050 * np.exp(-hip_roll_penalty):.2f} / 0.050\n"
            f"torque:  \t{0.025 * np.exp(-torque_penalty):.2f} / 0.025\n"
            f"action:  \t{0.025 * np.exp(-action_penalty):.2f} / 0.025"
        )

        print("actual speed: {}\tcommanded speed: {}\n\n".format(qvel[0], self.speed))
    return reward

def aslip_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    # foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)
    target_q = [1, 0, 0, 0]
    left_actual  = self.cassie_state.leftFoot.orientation
    right_actual = self.cassie_state.rightFoot.orientation
    foot_orient_error = 10 * ((1 - np.inner(left_actual, target_q) ** 2) + (1 - np.inner(right_actual, target_q) ** 2))

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - self.speed)
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors
    left_frc_clock = self.left_clock[0][0](self.phase)
    right_frc_clock = self.right_clock[0][0](self.phase)
    left_vel_clock = self.left_clock[0][1](self.phase)
    right_vel_clock = self.right_clock[0][1](self.phase)

    left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.25 * foot_frc_score +     \
                0.25 * foot_vel_score

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_frc_score:\t{4:.2f}, % = {5:.2f}\nfoot_vel_score:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),             0.2 * np.exp(-com_vel_error) / reward * 100,
        0.25 * foot_frc_score,                  0.25 * foot_frc_score / reward * 100,
        0.25 * foot_vel_score,                  0.25 * foot_vel_score / reward * 100,
        0.1  * np.exp(-straight_diff),            0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),         0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward

def max_vel_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_bonus     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 15 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)
    
    # com vel bonus
    com_vel_bonus += com_vel / 3.0

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (0 -> don't penalize), or penalize them (1 -> don't allow)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)
    
    left_frc_penalty = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_penalty = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_penalty = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_penalty = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_penalty = left_frc_penalty + right_frc_penalty
    foot_vel_penalty = left_vel_penalty + right_vel_penalty

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.2 * foot_frc_penalty +     \
                0.2 * foot_vel_penalty + \
                0.3 * com_vel_bonus
    
    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_penalty, foot_frc_penalty))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_penalty, foot_vel_penalty))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel_bonus:\t{2:.2f}, % = {3:.2f}\nfoot_frc_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_vel_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.3 * com_vel_bonus,                   0.3 * com_vel_bonus / reward * 100,
        0.2 * foot_frc_penalty,                0.2 * foot_frc_penalty / reward * 100,
        0.2 * foot_vel_penalty,                0.2 * foot_vel_penalty / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),      0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward