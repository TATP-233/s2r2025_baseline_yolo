import os
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy

from task_base import MMK2TaskBase
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):
    """
    模拟节点类，继承自MMK2TaskBase，实现具体的机器人控制逻辑
    Simulation node class, inherits from MMK2TaskBase, implements specific robot control logic
    """
    def __init__(self):
        """
        初始化SimNode类
        Initialize SimNode class
        """
        super().__init__()
        self.init_play()

    def init_play(self):
        """
        初始化任务执行相关变量和状态机
        Initialize task execution variables and state machine
        """
        self.stm = SimpleStateMachine()  # 创建简单状态机 | Create simple state machine
        self.stm.max_state_cnt = 25  # 动作流步数，根据任务需要调整 | Number of states in action flow, adjust according to task
        self.base_move_done = False  # 底盘移动是否完成 | Whether base movement is done
        self.delta_t = 1. / 24.  # 控制周期 | Control period

    def play_once(self):
        """
        执行一次完整的任务流程
        Execute one complete task flow
        """
        time.sleep(2.0)  # 等待系统稳定 | Wait for system to stabilize
        
        # 获取初始姿态 | Get initial pose
        init_pose = self.get_base_pose()
        self.target_posi_x = init_pose[0]  # 目标位置x坐标 | Target position x coordinate
        self.target_posi_y = init_pose[1]  # 目标位置y坐标 | Target position y coordinate
        self.target_yaw = init_pose[2]  # 目标偏航角 | Target yaw angle
        
        # 初始化各关节控制量为当前传感器读数 | Initialize joint control values to current sensor readings
        self.tctr_base[:2] = 0.0
        self.tctr_slide[:] = self.sensor_slide_qpos[:]
        self.tctr_head[:] = self.sensor_head_qpos[:]
        self.tctr_left_arm[:] = self.sensor_lft_arm_qpos[:]
        self.tctr_lft_gripper[:] = self.sensor_lft_gripper_qpos[:]
        self.tctr_right_arm[:] = self.sensor_rgt_arm_qpos[:]
        self.tctr_rgt_gripper[:] = self.sensor_rgt_gripper_qpos[:]
        
        # 设置动作初始值 | Set initial action values
        self.action[:] = self.target_control[:]

        # print("-" * 100)        
        # print("mj_data : ", self.mmk2_fk.mj_data.qpos)
        # print("self.tmat_cam_head =\n", self.tmat_cam_head)
        # print("-" * 100)
                
        # 创建控制循环的频率控制器 | Create rate controller for control loop
        rate = self.create_rate(int(1./self.delta_t))
        try:
            while rclpy.ok():
                # 处理状态机 | Process state machine
                self._process_state_machine()
                # 平滑过渡各关节的控制量 | Smooth transition of joint control values
                for i in range(2, len(self.target_control)):
                    self.action[i] = step_func(self.action[i], self.target_control[i], 1 * self.joint_move_ratio[i] * self.delta_t)
                # 更新控制量发送到机器人 | Update control values and send to robot
                self.updateControl()
                rate.sleep()
        except KeyboardInterrupt:
            print("shuting down ...")
        finally:
            print("action_done_dict : ")
            print(self.action_done_dict)

    def update_target_pose(self, target_name, z_range=[0.0, 2.0]):
        """
        基于YOLO检测更新目标位置
        Update target position based on YOLO detection
        
        Args:
            target_name: 目标名称 | Target name
            z_range: 有效的Z坐标范围 | Valid Z coordinate range
            
        Returns:
            目标在基座标系中的位置，如果未检测到则返回None | Target position in base frame, or None if not detected
        """
        print(f"检测到 {len(self.latest_detections)} 目标")
        if len(self.latest_detections) > 0:
            for i, obj in enumerate(self.latest_detections):
                print(i, "\n", obj)
                # 检查目标名称和Z坐标范围 | Check target name and Z coordinate range
                if obj['class'] == target_name and obj['Z'] > z_range[0] and obj['Z'] < z_range[1]:
                    # 提取基座标系中的位置 | Extract position in base frame
                    posi_local = np.array([obj['X'], obj['Y'], obj['Z']])
                    return posi_local
        else:
            print("未检测到目标")
            return None

    def _move_translation(self, translation):
        """
        控制机器人沿当前朝向平移指定距离
        Control robot to translate a specified distance along current orientation
        
        Args:
            translation: 平移距离，正值向前，负值向后 | Translation distance, positive for forward, negative for backward
        """
        # 控制频率 | Control frequency
        control_rate = 24
        rate = self.create_rate(control_rate)

        # 获取当前位姿 | Get current pose
        cur_pose = self.get_base_pose()
        ori_x, ori_y, ori_yaw = cur_pose[0], cur_pose[1], cur_pose[2]

        # 计算目标位置 | Calculate target position
        # 根据当前偏航角计算目标位置 | Calculate target position based on current yaw angle
        tar_x = cur_pose[0] + translation * np.cos(cur_pose[2])  # 目标x坐标 | Target x coordinate
        tar_y = cur_pose[1] + translation * np.sin(cur_pose[2])  # 目标y坐标 | Target y coordinate

        # 到达判断计数器 | Reach judgment counter
        reach_cnt = control_rate * 0.5
        for i in range(int(abs(translation) * 600)):
            # 获取最新位姿 | Get latest pose
            pose = self.get_base_pose()
            # 计算已移动距离和到目标的距离 | Calculate moved distance and distance to target
            move_dist = np.hypot(pose[0] - ori_x, pose[1] - ori_y)  # 已移动距离 | Distance moved
            tar_dist = np.hypot(pose[0] - tar_x, pose[1] - tar_y)   # 到目标的距离 | Distance to target
            # 检查是否失败或卡住 | Check if failed or stuck
            if (not rclpy.ok()) or (i > 40 and move_dist < 0.03):
                print(f"\033[0;31;40mbase_move: failed!\033[0m")
                return
            # 检查是否到达目标 | Check if reached target
            if tar_dist < 0.01 or move_dist > abs(translation) - 0.01:
                reach_cnt -= 1
                if reach_cnt < 0:
                    break
            # 计算线速度，与距离成正比 | Calculate linear velocity, proportional to distance
            target_lin_vel = np.clip(tar_dist * np.sign(translation) * 0.9, -0.65, 0.65)           
            self.base_move(target_lin_vel, 0.0)
            rate.sleep()

        # 停止移动 | Stop moving
        self.base_move(0.0, 0.0)
        print("base_move: translation done")
    
    def _move_rotate(self, rotation):
        """
        控制机器人旋转指定角度
        Control robot to rotate a specified angle
        
        Args:
            rotation: 旋转角度(弧度)，正值逆时针，负值顺时针 | Rotation angle (radians), positive for counterclockwise, negative for clockwise
        """
        # 控制频率 | Control frequency
        control_rate = 24
        rate = self.create_rate(control_rate)

        # 获取当前位姿 | Get current pose
        pose = self.get_base_pose()
        _, _, ori_yaw = pose[0], pose[1], pose[2]

        # 计算目标偏航角 | Calculate target yaw angle
        tar_yaw = self._std_angle(ori_yaw + rotation)

        # 到达判断计数器 | Reach judgment counter
        reach_cnt = control_rate * 0.5
        for i in range(int(abs(rotation) * 200)):
            # 获取最新位姿 | Get latest pose
            pose = self.get_base_pose()
            # 计算角度差 | Calculate angle difference
            ang_diff = self._angle_diff(tar_yaw, pose[2])
            # 检查是否失败或卡住 | Check if failed or stuck
            if (not rclpy.ok()) or (i > 40 and abs(self._angle_diff(pose[2], ori_yaw)) < 0.02):
                print(f"\033[0;31;40mbase_rotate: failed!\033[0m")
                return
            # 检查是否到达目标角度 | Check if reached target angle
            if abs(ang_diff) < 0.01:
                reach_cnt -= 1
                if reach_cnt < 0:
                    break
            # 角速度与角度差成正比 | Angular velocity proportional to angle difference
            self.base_move(0.0, np.clip(ang_diff, -0.8, 0.8) * 1.5)
            rate.sleep()

        # 停止旋转 | Stop rotating
        self.base_move(0.0, 0.0)
        print("rotate_control: rotation done")

    def _std_angle(self, ang):
        """
        标准化角度到[-π, π]范围
        Normalize angle to range [-π, π]
        
        Args:
            ang: 输入角度(弧度) | Input angle (radians)
            
        Returns:
            标准化后的角度(弧度) | Normalized angle (radians)
        """
        while ang > np.pi:
            ang -= 2*np.pi
        while ang < -np.pi:
            ang += 2*np.pi
        return ang

    def _angle_diff(self, a, b):
        """
        计算两个角度之间的最小差值
        Calculate minimum difference between two angles
        
        Args:
            a: 角度a(弧度) | Angle a (radians)
            b: 角度b(弧度) | Angle b (radians)
            
        Returns:
            最小角度差(弧度) | Minimum angle difference (radians)
        """
        diff = self._std_angle(a) - self._std_angle(b)
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def thread_moveto(self, target_x, target_y, target_yaw, forward_move=None):
        """
        创建线程移动到目标位置
        Create thread to move to target position
        
        Args:
            target_x: 目标x坐标 | Target x coordinate
            target_y: 目标y坐标 | Target y coordinate
            target_yaw: 目标偏航角 | Target yaw angle
            forward_move: 是否前进移动，None表示自动决定，True表示前进，False表示后退 | Whether to move forward, None for auto decide, True for forward, False for backward
        """
        # 自动决定前进或后退 | Automatically decide whether to move forward or backward
        if forward_move is None:
            cur_pose = self.get_base_pose()
            target_pose = np.array([target_x, target_y, target_yaw])

            # 计算目标方向与当前朝向的夹角 | Calculate angle between target direction and current orientation
            tar_ang = np.arctan2(target_pose[1] - cur_pose[1], target_pose[0] - cur_pose[0])
            rot_ang = self._angle_diff(tar_ang, cur_pose[2])

            # 如果目标在前方±90度范围内，前进，否则后退 | If target is within ±90 degrees in front, move forward, otherwise backward
            if abs(rot_ang) < np.pi/2.:
                self._moveto_directly(target_pose)
            else:
                self._movebackto_directly(target_pose)

        elif forward_move:
            self._moveto_directly()

        else:
            self._movebackto_directly()

        self.base_move_done = True

    def _moveto_directly(self, target):
        """
        直接移动到目标位置（前进）
        Move directly to target position (forward)
        
        Args:
            target: 目标位置 [x, y, yaw] | Target position [x, y, yaw]
        """
        # 获取当前位姿 | Get current pose
        pose = self.get_base_pose()
        # 计算目标方向 | Calculate target direction
        tar_ang = np.arctan2(target[1] - pose[1], target[0] - pose[0])
        # 计算需要旋转的角度 | Calculate rotation angle needed
        rot_ang = self._angle_diff(tar_ang, pose[2])
        # 先旋转到目标方向 | First rotate to target direction
        self._move_rotate(rot_ang)
        # 计算到目标的距离 | Calculate distance to target
        dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
        # 如果距离足够大，则平移 | If distance is large enough, translate
        if dist > 0.02:
            self._move_translation(dist)
        # 最后调整到目标朝向 | Finally adjust to target orientation
        for _ in range(3):
            pose = self.get_base_pose()
            rot_ang = self._angle_diff(target[2], pose[2])
            self._move_rotate(rot_ang)

    def _movebackto_directly(self, target):
        """
        直接移动到目标位置（后退）
        Move directly to target position (backward)
        
        Args:
            target: 目标位置 [x, y, yaw] | Target position [x, y, yaw]
        """
        # 获取当前位姿 | Get current pose
        pose = self.get_base_pose()
        # 计算反方向，即从目标到当前位置的方向 | Calculate opposite direction, i.e., from target to current position
        tar_ang = np.arctan2(pose[1]-target[1], pose[0]-target[0])
        # 计算需要旋转的角度 | Calculate rotation angle needed
        rot_ang = self._angle_diff(tar_ang, pose[2])
        # 先旋转到目标反方向 | First rotate to opposite direction of target
        self._move_rotate(rot_ang)
        # 计算到目标的距离 | Calculate distance to target
        dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
        # 如果距离足够大，则后退平移 | If distance is large enough, translate backward
        if dist > 0.02:
            self._move_translation(-dist)  # 负号表示后退 | Negative sign means backward
        # 最后调整到目标朝向 | Finally adjust to target orientation
        for _ in range(3):
            pose = self.get_base_pose()
            rot_ang = self._angle_diff(target[2], pose[2])
            self._move_rotate(rot_ang)

    def _process_state_machine(self):
        """
        处理状态机的状态转换和动作执行
        Process state machine state transitions and action execution
        """
        try:
            if self.stm.trigger():
                # 输出当前状态索引 | Output current state index
                print(f"self.stm.state_idx = {self.stm.state_idx}")
                
                # if self.stm.state_idx > 1:
                #     input(f"s{self.stm.state_idx} input to continue")

                # 根据状态索引执行不同的动作 | Execute different actions based on state index
                if self.stm.state_idx == 0:
                    # 状态0：移动到指定位置 | State 0: Move to specified position
                    self.base_move_done = False
                    self.target_posi_x = 0.55
                    self.target_posi_y = 0.52
                    self.target_yaw = 1.5708
                    # 创建移动线程 | Create movement thread
                    self.moveto_thead = threading.Thread(target=self.thread_moveto, args=(self.target_posi_x, self.target_posi_y, self.target_yaw))
                    self.moveto_thead.start()

                elif self.stm.state_idx == 1:
                    # 状态1：设置升降和头部位置以便观察 | State 1: Set slide and head position for observation
                    self.tctr_slide[0] = 0.7  # 设置升降高度 | Set slide height
                    self.tctr_head[0] = 0.   # 设置头部水平角度 | Set head horizontal angle
                    self.tctr_head[1] = 0.33  # 设置头部垂直角度 | Set head vertical angle
                    self.delay_cnt = int(2./self.delta_t)  # 延时2秒 | Delay 2 seconds

                elif self.stm.state_idx == 2:
                    # 状态2：检测盒子并准备抓取 | State 2: Detect box and prepare to grasp
                    # 检测盒子位置 | Detect box position
                    self.tar_box_posi = self.update_target_pose("box", [0.3, 0.7])

                    # 设置左臂位置为盒子上方 | Set left arm position above the box
                    tmp_lft_arm_target_pose = self.tar_box_posi + np.array([-0.15, 0.09, 0.06])
                    # 设置左臂末端姿态，使用ZYX欧拉角(0,0.8,π)表示旋转矩阵 | Set left arm end-effector pose using ZYX Euler angles (0,0.8,π) for rotation matrix
                    self.setArmEndTarget(tmp_lft_arm_target_pose, "carry", "l", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0.8,  np.pi]).as_matrix())
                    self.tctr_lft_gripper[:] = 1.0  # 打开左爪 | Open left gripper

                    # 设置右臂位置为盒子上方 | Set right arm position above the box
                    tmp_rgt_arm_target_pose = self.tar_box_posi + np.array([-0.15, -0.09, 0.06])
                    # 设置右臂末端姿态，使用ZYX欧拉角(0,0.8,-π)表示旋转矩阵 | Set right arm end-effector pose using ZYX Euler angles (0,0.8,-π) for rotation matrix
                    self.setArmEndTarget(tmp_rgt_arm_target_pose, "carry", "r", np.array(self.sensor_rgt_arm_qpos), Rotation.from_euler("zyx", [0., 0.8, -np.pi]).as_matrix())
                    self.tctr_rgt_gripper[:] = 1.0  # 打开右爪 | Open right gripper

                elif self.stm.state_idx == 3:
                    # 状态3：移动机械臂到盒子抓取位置 | State 3: Move arms to box grasping position
                    # 设置左臂接近盒子位置 | Set left arm closer to box
                    tmp_lft_arm_target_pose = self.tar_box_posi + np.array([0.03, 0.09, 0.01])
                    self.setArmEndTarget(tmp_lft_arm_target_pose, "carry", "l", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0.8,  np.pi]).as_matrix())

                    # 设置右臂接近盒子位置 | Set right arm closer to box
                    tmp_rgt_arm_target_pose = self.tar_box_posi + np.array([0.03, -0.09, 0.01])
                    self.setArmEndTarget(tmp_rgt_arm_target_pose, "carry", "r", np.array(self.sensor_rgt_arm_qpos), Rotation.from_euler("zyx", [0., 0.8, -np.pi]).as_matrix())

                elif self.stm.state_idx == 4:
                    # 状态4：夹紧盒子 | State 4: Grasp the box
                    # 关闭爪子抓住盒子 | Close grippers to grasp box
                    self.tctr_lft_gripper[:] = 0.05  # 关闭左爪 | Close left gripper
                    self.tctr_rgt_gripper[:] = 0.05  # 关闭右爪 | Close right gripper

                    self.delay_cnt = int(2./self.delta_t)  # 延时2秒 | Delay 2 seconds

                elif self.stm.state_idx == 5:
                    # 状态5：调整升降高度 | State 5: Adjust slide height
                    self.tctr_slide[0] = 0.67  # 稍微降低升降高度 | Slightly lower the slide height
                    self.tar_box_posi[2] += 0.03  # 盒子Z坐标上升，考虑到抓起后的高度变化 | Increase box Z coordinate, considering height change after grasping
                    
                elif self.stm.state_idx == 6:
                    # 状态6：调整机械臂位置准备搬运 | State 6: Adjust arm positions for carrying
                    # 调整左臂位置 | Adjust left arm position
                    tmp_lft_arm_target_pose = self.tar_box_posi + np.array([-0.09, 0.09, 0.025])
                    self.setArmEndTarget(tmp_lft_arm_target_pose, "carry", "l", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0.8,  np.pi]).as_matrix())

                    # 调整右臂位置 | Adjust right arm position
                    tmp_rgt_arm_target_pose = self.tar_box_posi + np.array([-0.09, -0.09, 0.025])
                    self.setArmEndTarget(tmp_rgt_arm_target_pose, "carry", "r", np.array(self.sensor_rgt_arm_qpos), Rotation.from_euler("zyx", [0., 0.8, -np.pi]).as_matrix())

                elif self.stm.state_idx == 7:
                    # 状态7：调整盒子位置到中间 | State 7: Adjust box position to center
                    self.tar_box_posi[1] = 0.0  # 将盒子Y坐标调整到中间 | Adjust box Y coordinate to center
                    
                    # 调整左臂位置 | Adjust left arm position
                    tmp_lft_arm_target_pose = self.tar_box_posi + np.array([-0.14, 0.09, 0.025])
                    self.setArmEndTarget(tmp_lft_arm_target_pose, "carry", "l", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0.8,  np.pi]).as_matrix())

                    # 调整右臂位置 | Adjust right arm position
                    tmp_rgt_arm_target_pose = self.tar_box_posi + np.array([-0.14, -0.09, 0.025])
                    self.setArmEndTarget(tmp_rgt_arm_target_pose, "carry", "r", np.array(self.sensor_rgt_arm_qpos), Rotation.from_euler("zyx", [0., 0.8, -np.pi]).as_matrix())

                elif self.stm.state_idx == 8:
                    # 状态8：移动到新位置 | State 8: Move to new position
                    self.base_move_done = False
                    self.target_posi_y = 0.3  # 调整Y坐标 | Adjust Y coordinate
                    # 创建移动线程 | Create movement thread
                    self.moveto_thead = threading.Thread(target=self.thread_moveto, args=(self.target_posi_x, self.target_posi_y, self.target_yaw))
                    self.moveto_thead.start()

                elif self.stm.state_idx == 9:
                    # 状态9：调整升降和头部位置 | State 9: Adjust slide and head positions
                    self.tctr_slide[0] = 0.  # 降低升降高度 | Lower slide height
                    self.tctr_head[0] = 0.  # 设置头部水平角度 | Set head horizontal angle
                    self.tctr_head[1] = 0.45  # 设置头部垂直角度 | Set head vertical angle
                    self.delay_cnt = int(2./self.delta_t)  # 延时2秒 | Delay 2 seconds

                elif self.stm.state_idx == 10:
                    # 状态10：移动到放置位置 | State 10: Move to placement position
                    self.base_move_done = False
                    self.target_posi_x = 0.  # 设置目标X坐标 | Set target X coordinate
                    self.target_posi_y = 0.05  # 设置目标Y坐标 | Set target Y coordinate
                    self.target_yaw = -1.5708  # 设置目标偏航角（-90度） | Set target yaw angle (-90 degrees)
                    # 创建移动线程 | Create movement thread
                    self.moveto_thead = threading.Thread(target=self.thread_moveto, args=(self.target_posi_x, self.target_posi_y, self.target_yaw))
                    self.moveto_thead.start()

                elif self.stm.state_idx == 11:
                    # 状态11：根据手的位置调整升降高度 | State 11: Adjust slide height based on hand positions
                    self.update_pose_fk()  # 更新正向运动学 | Update forward kinematics
                    # 获取左右手末端位置 | Get left and right hand end-effector positions
                    lft_hand_posi, _ = self.mmk2_fk.get_left_endeffector_pose()
                    rgt_hand_posi, _ = self.mmk2_fk.get_right_endeffector_pose()
                    # 计算双手高度平均值 | Calculate average height of both hands
                    hand_height = ((lft_hand_posi + rgt_hand_posi) * 0.5)[2]

                    # 设置升降高度，使手位于合适高度（0.87m基准） | Set slide height to position hands at appropriate height (0.87m reference)
                    self.tctr_slide[0] = hand_height - 0.87

                elif self.stm.state_idx == 12:
                    # 状态12：短暂延时 | State 12: Brief delay
                    self.delay_cnt = int(1./self.delta_t)  # 延时1秒 | Delay 1 second

                elif self.stm.state_idx == 13:
                    # 状态13：松开爪子放下盒子 | State 13: Open grippers to release box
                    self.tctr_lft_gripper[:] = 1.  # 打开左爪 | Open left gripper
                    self.tctr_rgt_gripper[:] = 1.  # 打开右爪 | Open right gripper
                    self.delay_cnt = int(2./self.delta_t)  # 延时2秒 | Delay 2 seconds

                elif self.stm.state_idx == 14:
                    # 状态14：降低升降高度 | State 14: Lower slide height
                    self.tctr_slide[0] = 0.  # 降低升降高度 | Lower slide height

                elif self.stm.state_idx == 15:                    
                    # 状态15：将双臂恢复到初始位置 | State 15: Return both arms to initial positions
                    # 设置左臂初始位置 | Set left arm to initial position
                    self.setArmEndTarget(self.arm_action_init_position[0], "pick", "l", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0., 0.]).as_matrix())
                    # 设置右臂初始位置 | Set right arm to initial position
                    self.setArmEndTarget(self.arm_action_init_position[1], "pick", "r", np.array(self.sensor_lft_arm_qpos), Rotation.from_euler("zyx", [0., 0., 0.]).as_matrix())

                elif self.stm.state_idx == 16:
                    # 状态16：移动到下一个任务位置 | State 16: Move to next task position
                    self.base_move_done = False
                    self.target_posi_x = 0.  # 设置目标X坐标 | Set target X coordinate
                    self.target_posi_y = -0.1  # 设置目标Y坐标 | Set target Y coordinate
                    self.target_yaw = -1.5708  # 设置目标偏航角（-90度） | Set target yaw angle (-90 degrees)
                    # 创建移动线程 | Create movement thread
                    self.moveto_thead = threading.Thread(target=self.thread_moveto, args=(self.target_posi_x, self.target_posi_y, self.target_yaw))
                    self.moveto_thead.start()

                elif self.stm.state_idx == 17:
                    # 状态17：抬高升降和头部以观察场景 | State 17: Raise slide and head to observe scene
                    self.tctr_slide[0] = 0.5  # 设置升降高度 | Set slide height
                    self.tctr_head[0] = 0.  # 设置头部水平角度 | Set head horizontal angle
                    self.tctr_head[1] = 0.5  # 设置头部垂直角度 | Set head vertical angle
                    self.delay_cnt = int(3./self.delta_t)  # 延时3秒 | Delay 3 seconds

                elif self.stm.state_idx == 18:
                    # 状态18：检测圆盘并调整姿态 | State 18: Detect disk and adjust posture
                    # 检测圆盘位置 | Detect disk position
                    self.tar_obj_posi = self.update_target_pose("disk", [0.75, 0.95])
                    print("-" * 100)
                    print(f"self.tar_obj_posi = {self.tar_obj_posi}")

                    # 调整升降和头部位置 | Adjust slide and head positions
                    self.tctr_slide[0] = 0.  # 降低升降高度 | Lower slide height
                    self.tctr_head[0] = 0.  # 设置头部水平角度 | Set head horizontal angle
                    self.tctr_head[1] = 0.55  # 设置头部垂直角度 | Set head vertical angle

                elif self.stm.state_idx == 19:
                    # 状态19：根据圆盘位置选择使用左臂或右臂抓取 | State 19: Choose left or right arm to grasp disk based on position
                    # 如果圆盘在左侧，使用左臂 | If disk is on the left side, use left arm
                    if self.tar_obj_posi[1] > 0.0:
                        # 设置左臂位置到圆盘上方 | Set left arm position above the disk
                        tmp_lft_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.1])
                        # 设置左臂末端姿态，使用单位矩阵表示垂直向下抓取 | Set left arm end-effector pose, using identity matrix for vertical downward grasping
                        self.setArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(self.sensor_lft_arm_qpos), np.eye(3))
                        self.tctr_lft_gripper[:] = 0.5  # 半开左爪 | Half-open left gripper
                    else:
                        # 设置右臂位置到圆盘上方 | Set right arm position above the disk
                        tmp_rgt_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.1])
                        # 设置右臂末端姿态，使用单位矩阵表示垂直向下抓取 | Set right arm end-effector pose, using identity matrix for vertical downward grasping
                        self.setArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(self.sensor_rgt_arm_qpos), np.eye(3))
                        self.tctr_rgt_gripper[:] = 0.5  # 半开右爪 | Half-open right gripper

                elif self.stm.state_idx == 20:
                    # 状态20：移动臂部到抓取位置 | State 20: Move arm to grasping position
                    # 如果圆盘在左侧，使用左臂 | If disk is on the left side, use left arm
                    if self.tar_obj_posi[1] > 0.0:
                        # 左臂下降到圆盘位置 | Lower left arm to disk position
                        tmp_lft_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.03])
                        self.setArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(self.sensor_lft_arm_qpos), np.eye(3))
                    else:
                        # 右臂下降到圆盘位置 | Lower right arm to disk position
                        tmp_rgt_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.03])
                        self.setArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(self.sensor_rgt_arm_qpos), np.eye(3))

                elif self.stm.state_idx == 21:
                    # 状态21：夹紧圆盘 | State 21: Grasp the disk
                    # 如果圆盘在左侧，使用左爪 | If disk is on the left side, use left gripper
                    if self.tar_obj_posi[1] > 0.0:
                        self.tctr_lft_gripper[:] = 0.0  # 关闭左爪 | Close left gripper
                    else:
                        self.tctr_rgt_gripper[:] = 0.0  # 关闭右爪 | Close right gripper
                    self.delay_cnt = int(1.5/self.delta_t)  # 延时1.5秒 | Delay 1.5 seconds

                elif self.stm.state_idx == 22:
                    # 状态22：提起圆盘 | State 22: Lift the disk
                    # 如果圆盘在左侧，使用左臂 | If disk is on the left side, use left arm
                    if self.tar_obj_posi[1] > 0.0:
                        # 左臂上升 | Raise left arm
                        tmp_lft_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.15])
                        self.setArmEndTarget(tmp_lft_arm_target_pose, "pick", "l", np.array(self.sensor_lft_arm_qpos), np.eye(3))
                    else:
                        # 右臂上升 | Raise right arm
                        tmp_rgt_arm_target_pose = self.tar_obj_posi + np.array([0., 0., 0.15])
                        self.setArmEndTarget(tmp_rgt_arm_target_pose, "pick", "r", np.array(self.sensor_rgt_arm_qpos), np.eye(3))

                elif self.stm.state_idx == 23:
                    # 状态23：移动到放置位置 | State 23: Move to placement position
                    self.base_move_done = False
                    self.target_posi_x = -0.5  # 设置目标X坐标 | Set target X coordinate
                    self.target_posi_y = -0.15  # 设置目标Y坐标 | Set target Y coordinate
                    self.target_yaw = -3.1416  # 设置目标偏航角（180度） | Set target yaw angle (180 degrees)
                    # 创建移动线程 | Create movement thread
                    self.moveto_thead = threading.Thread(target=self.thread_moveto, args=(self.target_posi_x, self.target_posi_y, self.target_yaw))
                    self.moveto_thead.start()

                elif self.stm.state_idx == 24:
                    # 状态24：放下圆盘 | State 24: Release the disk
                    # 如果圆盘在左侧，使用左爪 | If disk is on the left side, use left gripper
                    if self.tar_obj_posi[1] > 0.0:
                        self.tctr_lft_gripper[:] = 1.0  # 打开左爪 | Open left gripper
                    else:
                        self.tctr_rgt_gripper[:] = 1.0  # 打开右爪 | Open right gripper
                    self.delay_cnt = int(1.5/self.delta_t)  # 延时1.5秒 | Delay 1.5 seconds


                # 计算关节移动比率，用于平滑过渡 | Calculate joint movement ratio for smooth transition
                dif = np.abs(self.action - self.target_control)
                self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                self.joint_move_ratio[2] *= 0.15  # 降低升降关节的移动速度 | Reduce movement speed of slide joint

            else:
                # 更新状态机状态 | Update state machine state
                self.stm.update()

            # 如果底盘移动完成，微调偏航角以对准目标 | If base movement is done, fine-tune yaw angle to align with target
            if self.base_move_done:
                _, _, cur_yaw = self.get_base_pose()
                self.base_move(0, self._angle_diff(self.target_yaw, cur_yaw))
            
            # 检查所有动作是否完成，若完成则进入下一状态 | Check if all actions are done, if so move to next state
            if self.checkActionDone() and self.base_move_done:
                self.stm.next()
                print(f"stm.next() : {self.stm.state_idx}")

        except ValueError as ve:
            print(f"[ERROR] {ve}")

if __name__ == "__main__":
    # 设置NumPy打印选项 | Set NumPy print options
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    def run_yolo():
        """
        启动YOLO检测进程
        Start YOLO detection process
        """
        py_dir = os.popen('which python3').read().strip()
        yolo_py_dir = "/workspace/s2r2025_baseline_yolo/yolo_detect.py"
        os.system(f"{py_dir} {yolo_py_dir} --hide_info")
    
    # 创建并启动YOLO检测线程 | Create and start YOLO detection thread
    yolo_thread = threading.Thread(target=run_yolo)
    yolo_thread.start()
    
    # 初始化ROS | Initialize ROS
    rclpy.init()
    # 创建模拟节点 | Create simulation node
    mmk2_node = SimNode()

    # 创建并启动ROS消息处理线程 | Create and start ROS message processing thread
    spin_thead = threading.Thread(target=lambda: rclpy.spin(mmk2_node))
    spin_thead.start()

    # mmk2_node.stm.state_idx = 13
    # mmk2_node.stm.new_state = True
    # mmk2_node.base_move_done = True

    # 等待接收所有必要数据 | Wait for receiving all necessary data
    while rclpy.ok():
        if mmk2_node.recv_task_ and mmk2_node.recv_joint_states_ and mmk2_node.recv_odom_:
            print("all data received")
            break
        time.sleep(0.1)

    # 执行任务 | Execute task
    mmk2_node.play_once()

    # 等待线程结束并清理资源 | Wait for threads to end and clean up resources
    spin_thead.join()
    mmk2_node.destroy_node()
    rclpy.shutdown()

    yolo_thread.join()
