import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Float64MultiArray

from discoverse.mmk2 import MMK2FK, MMK2FIK

class MMK2TaskBase(Node):
    """
    MMK2机器人任务基类，提供基础功能接口
    Base class for MMK2 robot tasks, providing basic functionality interfaces
    """
    target_control = np.zeros(19)  # 目标控制量数组 | Target control values array
    action_done_dict = {
        "slide"         : False,  # 滑动关节是否完成 | Whether slide joint action is done
        "head"          : False,  # 头部关节是否完成 | Whether head joint action is done
        "left_arm"      : False,  # 左臂关节是否完成 | Whether left arm action is done
        "left_gripper"  : False,  # 左手爪是否完成 | Whether left gripper action is done
        "right_arm"     : False,  # 右臂关节是否完成 | Whether right arm action is done
        "right_gripper" : False,  # 右手爪是否完成 | Whether right gripper action is done
        "delay"         : False,  # 延时是否完成 | Whether delay is done
    }
    delay_cnt = 0  # 延时计数器 | Delay counter
    set_left_arm_new_target = False  # 是否设置左臂新目标 | Whether to set new target for left arm
    set_right_arm_new_target = False  # 是否设置右臂新目标 | Whether to set new target for right arm

    def __init__(self):
        """
        初始化MMK2TaskBase类
        Initialize MMK2TaskBase class
        """
        super().__init__('mmk2_node')
        self.mmk2_fk = MMK2FK()  # 初始化MMK2正向运动学 | Initialize MMK2 forward kinematics
        
        self.arm_action = "pick"  # 设置臂部动作类型 | Set arm action type
        # 将目标控制量数组分割为各个子系统 | Split target control array into subsystems
        self.tctr_base = self.target_control[:2]  # 底盘控制 | Base control
        self.tctr_slide = self.target_control[2:3]  # 升降控制 | Slide control
        self.tctr_head = self.target_control[3:5]  # 头部控制 | Head control
        self.tctr_left_arm = self.target_control[5:11]  # 左臂控制 | Left arm control
        self.tctr_lft_gripper = self.target_control[11:12]  # 左爪控制 | Left gripper control
        self.tctr_right_arm = self.target_control[12:18]  # 右臂控制 | Right arm control
        self.tctr_rgt_gripper = self.target_control[18:19]  # 右爪控制 | Right gripper control
        self.action = np.zeros_like(self.target_control)  # 实际执行动作 | Actual action to execute        

        self.recv_task_ = False  # 是否接收到任务 | Whether task is received
        self.task_info = None  # 任务信息 | Task information
        self.recv_odom_ = False  # 是否接收到里程计数据 | Whether odometry data is received
        self.recv_joint_states_ = False  # 是否接收到关节状态 | Whether joint states are received
        self.init_jotnt_seq_ = False
        
        # 观测状态 | Observation state
        self.obs = {
            "time": None,  # 时间 | Time
            "jq": np.array([0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0.]),  # 关节角度 | Joint angles
            "base_position": [0., 0., 0.],  # 底盘位置 | Base position
            "base_orientation": [1., 0., 0., 0.],  # 底盘方向（四元数w,x,y,z） | Base orientation (quaternion w,x,y,z)
        }
        self.joint_seq = np.zeros_like(self.obs["jq"], dtype=np.int32)

        self.init_subscription()  # 初始化订阅 | Initialize subscriptions
        self.init_publisher()  # 初始化发布器 | Initialize publishers
        self.initial_pose = [0,0,0]  # 初始姿态 | Initial pose
        # 臂部动作初始位置 | Arm action initial positions
        self.arm_action_init_position = np.array([
            [0.223,  0.21, 1.07055],  # 左臂初始位置 | Left arm initial position
            [0.223, -0.21, 1.07055],  # 右臂初始位置 | Right arm initial position
        ])

        self.latest_detections = []  # 最新检测结果 | Latest detection results

    def get_tmat_wrt_mmk2base(self, pose):
        """
        获取相对于MMK2基座标系的变换矩阵
        Get transformation matrix with respect to MMK2 base frame
        
        Args:
            pose: 世界坐标系中的位置 | Position in world frame
            
        Returns:
            相对于MMK2基座标系的位置 | Position with respect to MMK2 base frame
        """
        current_pos  = self.obs["base_position"]   # 当前位置[X, Y, Z] | Current position [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # 当前四元数[qw, qx, qy, qz] | Current quaternion [qw, qx, qy, qz]
        # 创建MMK2在世界坐标系中的变换矩阵 | Create transformation matrix of MMK2 in world frame
        tmat_mmk2 = np.eye(4)  # 单位矩阵 | Identity matrix
        # 四元数转旋转矩阵，注意四元数顺序调整为[qx, qy, qz, qw] | Convert quaternion to rotation matrix, note quaternion order is adjusted to [qx, qy, qz, qw]
        tmat_mmk2[:3,:3] = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_matrix()
        tmat_mmk2[:3,3] = current_pos  # 设置平移部分 | Set translation part
        # 计算世界坐标系到MMK2基座标系的变换 | Calculate transformation from world frame to MMK2 base frame
        return (np.linalg.inv(tmat_mmk2) @ np.append(pose, 1))[:3]

    def setArmEndTarget(self, target_pose, arm_action, arm, q_ref, a_rot):
        """
        设置机械臂末端目标位置
        Set arm end-effector target position
        
        Args:
            target_pose: 目标位置 | Target position
            arm_action: 臂部动作类型 | Arm action type
            arm: 使用哪个臂 ('l'为左臂, 'r'为右臂) | Which arm to use ('l' for left, 'r' for right)
            q_ref: 参考关节角度 | Reference joint angles
            a_rot: 目标旋转矩阵 | Target rotation matrix
            
        Returns:
            是否成功 | Whether successful
        """
        # print(f"Setting {arm.upper()} arm target local : {np.array2string(target_pose, separator=', ')}")
    
        try:
            # 求解逆运动学获取关节角度 | Solve inverse kinematics to get joint angles
            # 在机器人底盘坐标系中计算 | Calculate in robot footprint frame
            rq = MMK2FIK().get_armjoint_pose_wrt_footprint(target_pose, arm_action, arm, self.tctr_slide[0], q_ref, a_rot)
            if arm == "l":
                self.tctr_left_arm[:] = rq  # 设置左臂目标关节角度 | Set left arm target joint angles
                self.set_left_arm_new_target = True
            elif arm == "r":
                self.tctr_right_arm[:] = rq  # 设置右臂目标关节角度 | Set right arm target joint angles
                self.set_right_arm_new_target = True
            return True
        
        except ValueError as e:
            print(f"Failed to solve IK {e} params: arm={arm}, target={target_pose}, slide={self.tctr_slide[0]:.2f}")
            return False

    def base_move(self, vx, vyaw):
        """
        底盘移动控制
        Base movement control
        
        Args:
            vx: x方向线速度 | Linear velocity in x direction
            vyaw: 偏航角速度 | Yaw angular velocity
        """
        self.action[0] = vx  # 设置x方向线速度 | Set linear velocity in x direction
        self.action[1] = vyaw  # 设置偏航角速度 | Set yaw angular velocity
    
    def updateControl(self):
        """
        更新各个关节的控制量并发布
        Update control values for all joints and publish
        """
        # 发布底盘速度命令 | Publish base velocity command
        twist_msg = Twist()
        twist_msg.linear.x = self.action[0]
        twist_msg.angular.z = self.action[1]
        self.publisher_cmd_vel.publish(twist_msg)

        # 发布升降关节命令 | Publish slide joint command
        float_array_msg_spine = Float64MultiArray()
        float_array_msg_spine.data = self.action[2:3].tolist()
        self.publisher_spine.publish(float_array_msg_spine)

        # 发布头部关节命令 | Publish head joint command
        float_array_msg_head = Float64MultiArray()
        float_array_msg_head.data = self.action[3:5].tolist()
        self.publisher_head.publish(float_array_msg_head)

        # 发布左臂关节命令 | Publish left arm joint command
        float_array_msg_left_arm = Float64MultiArray()
        float_array_msg_left_arm.data = self.action[5:12].tolist()
        self.publisher_left_arm.publish(float_array_msg_left_arm)

        # 发布右臂关节命令 | Publish right arm joint command
        float_array_msg_right_arm = Float64MultiArray()
        float_array_msg_right_arm.data = self.action[12:19].tolist()
        self.publisher_right_arm.publish(float_array_msg_right_arm)

    def get_base_pose(self):
        """
        获取底盘当前位姿 [x, y, yaw]
        Get current base pose [x, y, yaw]
        
        Returns:
            底盘位姿 [x, y, yaw] | Base pose [x, y, yaw]
        """
        current_pos  = self.obs["base_position"]   # 当前位置[X, Y, Z] | Current position [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # 当前四元数[qw, qx, qy, qz] | Current quaternion [qw, qx, qy, qz]
        # 四元数转欧拉角，获取偏航角 | Convert quaternion to Euler angles, get yaw angle
        yaw = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_euler('zyx')[0]
        return np.array([current_pos[0], current_pos[1], yaw])

    def checkActionDone(self, debug=False):
        """
        检查各个动作是否完成
        Check if all actions are done
        
        Args:
            debug: 是否输出调试信息 | Whether to output debug information
            
        Returns:
            所有动作是否完成 | Whether all actions are done
        """
        # 检查各个关节是否到达目标位置 | Check if each joint has reached its target position
        slide_done = np.allclose(self.tctr_slide, self.sensor_slide_qpos, atol=3e-2)
        head_done  = np.allclose(self.tctr_head,  self.sensor_head_qpos,  atol=3e-2)

        # 检查左臂是否到达目标位置 | Check if left arm has reached its target position
        if self.set_left_arm_new_target:
            left_arm_done = np.allclose(self.tctr_left_arm, self.sensor_lft_arm_qpos, atol=5e-2)
            if left_arm_done:
                self.set_left_arm_new_target = False
        else:
            left_arm_done = True

        # 检查右臂是否到达目标位置 | Check if right arm has reached its target position
        if self.set_right_arm_new_target:
            right_arm_done = np.allclose(self.tctr_right_arm, self.sensor_rgt_arm_qpos, atol=5e-2)
            if right_arm_done:
                self.set_right_arm_new_target = False
        else:
            right_arm_done = True

        # 检查爪子是否到达目标位置 | Check if grippers have reached their target positions
        left_gripper_done  = np.allclose(self.tctr_lft_gripper, self.sensor_lft_gripper_qpos, atol=0.5)
        right_gripper_done = np.allclose(self.tctr_rgt_gripper, self.sensor_rgt_gripper_qpos, atol=0.5)

        # 检查延时是否完成 | Check if delay is done
        self.delay_cnt -= 1
        delay_done = (self.delay_cnt<=0)

        # 更新动作完成状态字典 | Update action done status dictionary
        self.action_done_dict = {
            "slide"         : slide_done,
            "head"          : head_done,
            "left_arm"      : left_arm_done,
            "left_gripper"  : left_gripper_done,
            "right_arm"     : right_arm_done,
            "right_gripper" : right_gripper_done,
            "delay"         : delay_done,
        }

        # 返回所有动作是否完成 | Return whether all actions are done
        return slide_done and head_done and left_arm_done and left_gripper_done and right_arm_done and right_gripper_done and delay_done

    def init_subscription(self):
        """
        初始化所有ROS话题订阅
        Initialize all ROS topic subscriptions
        """
        self.sub_odom = self.create_subscription(Odometry, '/slamware_ros_sdk_server_node/odom', self.odom_callback, 10)
        self.sub_joint_states = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.sub_taskinfo = self.create_subscription(String, '/s2r2025/taskinfo', self.taskinfo_callback, 10)
        self.subscription_gameinfo = self.create_subscription(String, '/s2r2025/gameinfo', self.gameinfo_callback, 10)
        self.sub_detect = self.create_subscription(Detection2DArray, '/yolo/detections', self._detection_callback, 10)

    def init_publisher(self):
        """
        初始化所有ROS话题发布器
        Initialize all ROS topic publishers
        """
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher_head = self.create_publisher(Float64MultiArray, '/head_forward_position_controller/commands', 10)
        self.publisher_left_arm = self.create_publisher(Float64MultiArray, '/left_arm_forward_position_controller/commands', 10)
        self.publisher_right_arm = self.create_publisher(Float64MultiArray, '/right_arm_forward_position_controller/commands', 10)
        self.publisher_spine = self.create_publisher(Float64MultiArray, '/spine_forward_position_controller/commands', 10)
    
    def _detection_callback(self, msg):
        """
        YOLO检测结果回调函数
        Callback function for YOLO detection results
        
        Args:
            msg: 检测结果消息 | Detection result message
        """
        current_detections = []

        if not msg.detections:
            self.latest_detections = []
            return

        try:
            self.update_pose_fk()
        except AttributeError:
            return

        # 获取头部相机的位姿 | Get head camera pose
        cam_head_trans, cam_head_ori = self.mmk2_fk.get_head_camera_pose()
        # 创建头部相机在MMK2基座标系中的变换矩阵 | Create transformation matrix for head camera in MMK2 base frame
        self.tmat_cam_head = np.eye(4)
        self.tmat_cam_head[:3, 3] = cam_head_trans
        # 四元数转旋转矩阵，注意四元数顺序调整 | Convert quaternion to rotation matrix, adjust quaternion order
        self.tmat_cam_head[:3, :3] = Rotation.from_quat(cam_head_ori[[1,2,3,0]]).as_matrix()

        for detection in msg.detections:
            # 获取相机坐标系中的物体位置 | Get object position in camera frame
            obj_cam_pose = [detection.results[0].pose.pose.position.x,
                            detection.results[0].pose.pose.position.y,
                            detection.results[0].pose.pose.position.z ]
            # 将物体位置从相机坐标系转换到基座标系 | Convert object position from camera frame to base frame
            obj_base_pose = self.pcam2base(obj_cam_pose)  # 相对于基座标系 | relative to base frame

            # ROS2 Humble
            obj = {
                'class': detection.results[0].hypothesis.class_id,  # 物体类别 | Object class
                'confidence': detection.results[0].hypothesis.score,  # 置信度 | Confidence score
                'x': detection.bbox.center.position.x,  # 图像中x坐标 | x coordinate in image
                'y': detection.bbox.center.position.y,  # 图像中y坐标 | y coordinate in image
                'w': detection.bbox.size_x,  # 图像中宽度 | width in image
                'h': detection.bbox.size_y,  # 图像中高度 | height in image
                'X': round(obj_base_pose[0], 4),  # 基座标系中X坐标 | X coordinate in base frame
                'Y': round(obj_base_pose[1], 4),  # 基座标系中Y坐标 | Y coordinate in base frame
                'Z': round(obj_base_pose[2], 4)   # 基座标系中Z坐标 | Z coordinate in base frame
            }
            current_detections.append(obj)
        self.latest_detections = current_detections

    def update_pose_fk(self):
        """
        更新正向运动学模型中的关节角度
        Update joint angles in forward kinematics model
        """
        self.mmk2_fk.set_base_pose(self.obs["base_position"], self.obs["base_orientation"])
        self.mmk2_fk.set_slide_joint(self.sensor_slide_qpos[0])
        self.mmk2_fk.set_head_joints(self.sensor_head_qpos)
        self.mmk2_fk.set_left_arm_joints(self.sensor_lft_arm_qpos)
        self.mmk2_fk.set_right_arm_joints(self.sensor_rgt_arm_qpos)

    def pcam2base(self, target):
        """
        将目标点从相机坐标系转换到机器人基座标系
        Convert target point from camera frame to robot base frame
        
        Args:
            target: 相机坐标系中的点 [X, Y, Z] | Point in camera frame [X, Y, Z]
            
        Returns:
            机器人基座标系中的点 [X, Y, Z] | Point in robot base frame [X, Y, Z]
        """
        # 转换为齐次坐标 | Convert to homogeneous coordinates
        point3d = np.array([target[0], target[1], target[2], 1.0])
        # 应用相机到世界的变换 | Apply camera to world transformation
        posi_world = self.tmat_cam_head @ point3d

        # 获取MMK2在世界坐标系中的位姿 | Get MMK2 pose in world frame
        current_pos  = self.obs["base_position"]   # 当前位置[X, Y, Z] | Current position [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # 当前四元数[qw, qx, qy, qz] | Current quaternion [qw, qx, qy, qz]

        # 创建MMK2在世界坐标系中的变换矩阵 | Create transformation matrix of MMK2 in world frame
        tmat_mmk2 = np.eye(4)
        tmat_mmk2[:3,3]  = current_pos
        # 四元数转旋转矩阵 | Convert quaternion to rotation matrix
        tmat_mmk2[:3,:3] = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]]).as_matrix()
        
        # 世界坐标系到MMK2基座标系的变换 | Transform from world frame to MMK2 base frame
        posi_local = (np.linalg.inv(tmat_mmk2) @ posi_world)[:3]

        return posi_local

    def odom_callback(self, msg):
        """
        里程计数据回调函数
        Odometry data callback function
        
        Args:
            msg: 里程计消息 | Odometry message
        """
        # 提取位置信息 | Extract position information
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        self.obs["base_position"] = [position_x, position_y, position_z]

        # 提取方向信息（四元数） | Extract orientation information (quaternion)
        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        self.obs["base_orientation"] = [orientation_w, orientation_x, orientation_y, orientation_z]
        self.recv_odom_ = True

    def joint_states_callback(self, msg:JointState):
        """
        关节状态回调函数
        Joint states callback function
        
        Args:
            msg: 关节状态消息 | Joint states message
        """
        if not self.init_jotnt_seq_:
            joint_names = [
                "slide_joint", "head_yaw_joint", "head_pitch_joint",
                "left_arm_joint1" , "left_arm_joint2" , "left_arm_joint3" , "left_arm_joint4" , "left_arm_joint5" , "left_arm_joint6" , "left_arm_eef_gripper_joint" ,
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
            ]

            if len(msg.name) != len(joint_names):
                print(f"Joint names length mismatch: {len(msg.name)} != {len(joint_names)}")
                return 

            msg_name = list(msg.name)
            for i, n in enumerate(joint_names):
                try:
                    self.joint_seq[i] = msg_name.index(n)
                except ValueError:
                    print(f"Joint name {n} not found in message")
                    return
            print(f"Joint sequence: {self.joint_seq}")
            self.init_jotnt_seq_ = True            

        self.obs["jq"] = np.array(msg.position)
        # 分割关节角度到各个子系统 | Split joint angles to subsystems
        self.sensor_slide_qpos = self.obs["jq"][self.joint_seq[:1]]  # 升降关节角度 | Slide joint angle
        self.sensor_head_qpos  = self.obs["jq"][self.joint_seq[1:3]]  # 头部关节角度 | Head joint angles
        self.sensor_lft_arm_qpos  = self.obs["jq"][self.joint_seq[3:9]]  # 左臂关节角度 | Left arm joint angles
        self.sensor_lft_gripper_qpos  = self.obs["jq"][self.joint_seq[9:10]]  # 左爪关节角度 | Left gripper joint angle
        self.sensor_rgt_arm_qpos  = self.obs["jq"][self.joint_seq[10:16]]  # 右臂关节角度 | Right arm joint angles
        self.sensor_rgt_gripper_qpos  = self.obs["jq"][self.joint_seq[16:17]]  # 右爪关节角度 | Right gripper joint angle
        self.recv_joint_states_ = True

    def taskinfo_callback(self, msg):
        """
        任务信息回调函数
        Task information callback function
        
        Args:
            msg: 任务信息消息 | Task information message
        """
        self.task_info = msg.data
        self.recv_task_ = True

    def gameinfo_callback(self, msg):
        """
        游戏信息回调函数
        Game information callback function
        
        Args:
            msg: 游戏信息消息 | Game information message
        """
        pass

if __name__ == '__main__':
    rclpy.init()
    mmk2_node = MMK2TaskBase()
    # 创建并启动ROS消息处理线程 | Create and start ROS message processing thread
    spin_thead = threading.Thread(target=lambda: rclpy.spin(mmk2_node))
    spin_thead.start()
    # 创建并启动控制发布线程 | Create and start control publishing thread
    pub_thread = threading.Thread(target=mmk2_node.pub_thread)
    pub_thread.start()

    # 等待线程结束 | Wait for threads to finish
    spin_thead.join()
    pub_thread.join()
    # 清理资源 | Clean up resources
    mmk2_node.destroy_node()
    rclpy.shutdown()