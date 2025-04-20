'''
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-03-14 20:41:48
LastEditTime: 2025-03-18 12:54:25
'''
import os
import cv2
import torch
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

USE_ULTRALYTICS = False

if USE_ULTRALYTICS:
    from ultralytics import YOLO
else:
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords

# 为每个类别随机分配颜色，用于可视化 | Randomly assign colors for each class for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (255, 128, 128), (128, 0, 128), (0, 0, 0)
]

class YOLODetectorNode(Node):
    """
    YOLO目标检测节点，负责检测图像中的物体并发布结果
    YOLO object detection node, responsible for detecting objects in images and publishing results
    """
    def __init__(self, print_log=False, pub_res_img=False, verbose=False):
        """
        初始化YOLO检测器节点
        Initialize YOLO detector node
        
        Args:
            verbose: 是否输出详细信息 | Whether to output detailed information
        """
        super().__init__('yolo_detector')
        self.bridge = CvBridge()  # 用于转换ROS图像和OpenCV图像 | For converting between ROS images and OpenCV images
        self.camera_intrinsic = None  # 相机内参 | Camera intrinsic parameters
        self.camera_distCoeffs = None  # 相机畸变系数 | Camera distortion coefficients
        self._depth_msg = None  # 最新的深度图像 | Latest depth image
        self.print_log = print_log  # 是否输出详细信息 | Whether to output detailed information
        self.pub_result_image = pub_res_img
        self.verbose = verbose

        if self.verbose:
            self.window_name = "yolo_detect"
            cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
            cv2.setMouseCallback(self.window_name, self.mouseCallback)
            self.split_line = 640
            self.mouse_x = 0
            self.mouse_y = 0

        # 初始化YOLO模型、ROS话题订阅和发布 | Initialize YOLO model, ROS topic subscriptions and publishers
        if USE_ULTRALYTICS:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/prize_m.pt')
        else:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/best.pt')
        self.init_model(model_path, USE_ULTRALYTICS)
        self.init_subscription()
        self.init_publisher()

    def init_model(self, model_path, use_ultralytics=True):
        """
        初始化YOLO模型
        Initialize YOLO model
        
        Args:
            model_path: YOLO模型文件路径 | Path to YOLO model file
        """
        # 设置设备（GPU或CPU） | Set device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_ultralytics:
            from ultralytics import YOLO
            try:
                # 加载YOLO模型 | Load YOLO model
                self.model = YOLO(model_path).to(self.device)
            except Exception as e:
                raise RuntimeError(f"failed to load yolo model: {str(e)}")
        else:
            self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
            if self.device.type != 'cpu':
                self.half_model = True
                self.model.half()  # to FP16
                img_size = 640
                self.model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
            self.stride = int(self.model.stride.max())  # model stride

            color_img = np.zeros((640, 640, 3), np.uint8)
            s = np.stack([letterbox(color_img, img_size, stride=self.stride)[0].shape], 0)  # shapes
            self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

        # 设置模型为评估模式 | Set model to evaluation mode
        self.model.eval()
        self.img_size = 640  # 图像大小 | Image size
        self.conf_thresh = 0.75  # 置信度阈值 | Confidence threshold
        # 定义识别的物体类别 | Define object classes to be recognized
        self.class_names = ["box", "carton", "disk", "sheet", "airbot", "blue_circle", "-1"]

    def init_subscription(self):
        """
        初始化ROS话题订阅
        Initialize ROS topic subscriptions
        """
        # 订阅相机内参话题 | Subscribe to camera intrinsic parameters topic
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/head_camera/color/camera_info',
            self.camera_info_callback, 10)
        
        # 订阅深度图像话题 | Subscribe to depth image topic
        self.depth_sub = self.create_subscription(
            Image, '/head_camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10)
        
        # 订阅RGB图像话题 | Subscribe to RGB image topic
        self.rgb_sub = self.create_subscription(
            Image, '/head_camera/color/image_raw',
            self.rgb_callback, 10)

    def init_publisher(self):
        """
        初始化ROS话题发布器
        Initialize ROS topic publishers
        """
        # 发布检测结果话题 | Publish detection results topic
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)

        if self.pub_result_image or self.verbose:
            # 发布可视化结果图像话题 | Publish visualized result image topic
            self.result_pub = self.create_publisher(Image, '/yolo/result_image', 10)

    def camera_info_callback(self, msg:CameraInfo):
        """
        相机内参信息回调函数
        Camera intrinsic information callback function
        
        Args:
            msg: 相机内参消息 | Camera intrinsic message
        """
        # 提取相机畸变系数 | Extract camera distortion coefficients
        if len(msg.d):
            self.camera_distCoeffs = np.array(msg.d).flatten()
        # 提取相机内参矩阵 | Extract camera intrinsic matrix
        self.camera_intrinsic = np.array(msg.k).reshape(3,3)

    def mouseCallback(self, event, x, y, flags, param):
        _x, _y, width, height = cv2.getWindowImageRect(self.window_name)
        if flags == cv2.EVENT_FLAG_LBUTTON:
            self.split_line = min(max(x, 0), width)
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = min(max(x, 0), width-1)
            self.mouse_y = min(max(y, 0), height-1)

    def depth_callback(self, msg:Image):
        """
        深度图像回调函数
        Depth image callback function
        
        Args:
            msg: 深度图像消息 | Depth image message
        """
        # 将ROS深度图像转换为OpenCV格式 | Convert ROS depth image to OpenCV format
        self._depth_msg = msg

    def rgb_callback(self, msg:Image):
        """
        RGB图像回调函数
        RGB image callback function
        
        Args:
            msg: RGB图像消息 | RGB image message
        """
        # 检查是否已接收到相机内参和深度图像 | Check if camera intrinsic and depth image have been received
        if self.camera_intrinsic is None or self._depth_msg is None:
            return
        
        # 将ROS RGB图像转换为OpenCV格式 | Convert ROS RGB image to OpenCV format
        rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(self._depth_msg)
        
        # 检测物体并生成可视化图像 | Detect objects and generate visualized image
        visualized_img, detected_objects = self.detect_objects(
            rgb_image, 
            depth_image,
            self.camera_intrinsic,
            self.camera_distCoeffs
        )
        
        # 发布检测结果和可视化图像 | Publish detection results and visualized image
        self.publish_detections(detected_objects)

        if self.pub_result_image:
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(visualized_img, 'bgr8'))

        if self.verbose:
            # check 窗口是否还存在 | Check if window still exists
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, exiting...")
                raise KeyboardInterrupt

            # 在窗口中显示RGB图像 | Display RGB image in window
            depth = depth_image[self.mouse_y, self.mouse_x] * 1e-3
            point = self.pixel2world(depth, self.mouse_x, self.mouse_y, self.camera_intrinsic, self.camera_distCoeffs)
            cv2.putText(visualized_img, f"({point[0]:4.2f},{point[1]:4.2f},{point[2]:4.2f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 在窗口中显示检测结果 | Display detection results in window
            cv2.imshow(self.window_name, visualized_img)
            key = cv2.waitKey(1)
            if key == 27:
                print("ESC key pressed, exiting...")
                raise KeyboardInterrupt

    def detect_objects(self, rgb_image, depth_image, camera_intrinsic, camera_distCoeffs):
        """
        使用YOLO模型检测物体，并将2D检测结果转换为3D位置
        Detect objects using YOLO model and convert 2D detection results to 3D positions
        
        Args:
            rgb_image: RGB图像 | RGB image
            img_depth: 深度图像 (毫米) | Depth image (mm)
            camera_intrinsic: 相机内参矩阵 | Camera intrinsic matrix
            
        Returns:
            带有标注的图像和检测到的物体列表 | Annotated image and list of detected objects
        """
        # 使用YOLO模型进行目标检测 | Use YOLO model for object detection
        if USE_ULTRALYTICS:
            results = self.model(rgb_image, verbose=self.print_log)[0]
        else:
            im0 = rgb_image.copy()

            img_size = 640
            img = np.stack([letterbox(im0, img_size, auto=self.rect, stride=self.stride)[0]], 0)

            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half_model else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = self.model(img)[0]

            results = non_max_suppression(pred, self.conf_thresh, 0.45)[0]
            if len(results):
                results[:, :4] = scale_coords(img.shape[2:], results[:, :4], im0.shape).round()

        # 后处理检测结果 | Post-process detection results
        if self.verbose:
            depth_img_color = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth_image, 0.0, 3000.0), alpha=0.255/3.), cv2.COLORMAP_JET)
            rgb_image[:, self.split_line:, :] = depth_img_color[:, self.split_line:, :]

        image, detected_objects = self.postprocess(results, rgb_image)
        
        # 遍历检测到的每个物体 | For each detected object
        for obj in detected_objects:
            obj_pix_x = obj['x']  # 物体在图像中的x坐标 | Object x coordinate in image
            obj_pix_y = obj['y']  # 物体在图像中的y坐标 | Object y coordinate in image
            depth_m = depth_image[obj_pix_y, obj_pix_x] * 1e-3  # 将深度从mm转换为m | Convert depth from mm to m

            # 计算物体在相机坐标系中的3D位置 | Calculate object 3D position in camera coordinate system
            point_xyz = self.pixel2world(depth_m, obj_pix_x, obj_pix_y, camera_intrinsic, camera_distCoeffs)
            obj.update({
                'X': round(point_xyz[0], 3),  # 相机坐标系中的X坐标 | X coordinate in camera frame
                'Y': round(point_xyz[1], 3),  # 相机坐标系中的Y坐标 | Y coordinate in camera frame
                'Z': round(point_xyz[2], 3)   # 相机坐标系中的Z坐标 | Z coordinate in camera frame
            })

            # 在图像上绘制物体位置信息 | Draw object position information on image
            if self.pub_result_image or self.verbose:
                color = COLORS[self.class_names.index(obj['class'])]
                cv2.putText(image, f"{obj['class']}: ({point_xyz[0]:.2f},{point_xyz[1]:.2f},{point_xyz[2]:.2f})m", 
                            (obj_pix_x - 50, obj_pix_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 1)

        return image, detected_objects

    def pixel2world(self, depth_in_meters, pix_x, pix_y, intrinsic, camera_distCoeffs=None):
        """
        camera_distCoeffs: Input vector of distortion coefficients 
            shape (5, ) : (k1, k2, P1, P2, K3)
            shape (8, ) : (k1, k2, P1, P2, K3, k4, k5, k6)
        """
        # 计算相机坐标系中的3D点 | Calculate 3D point in camera coordinate system
        if camera_distCoeffs is None:
            # 从相机内参中提取参数 | Extract parameters from camera intrinsics
            fx = intrinsic[0,0]  # x方向焦距 | Focal length in x direction
            fy = intrinsic[1,1]  # y方向焦距 | Focal length in y direction
            cx = intrinsic[0,2]  # x方向光心 | Principal point in x direction
            cy = intrinsic[1,2]  # y方向光心 | Principal point in y direction
            x = (pix_x - cx) * depth_in_meters / fx
            y = (pix_y - cy) * depth_in_meters / fy
        else:
            udpixel = cv2.undistortPoints(np.array([[pix_x, pix_y]], np.float32), intrinsic, camera_distCoeffs)
            x = udpixel.flatten()[0]
            y = udpixel.flatten()[1]
        return np.array([x, y, 1.0]) * depth_in_meters

    def postprocess(self, results, orig_img):
        """
        后处理YOLO检测结果
        Post-process YOLO detection results
        
        Args:
            results: YOLO检测结果 | YOLO detection results
            orig_img: 原始图像 | Original image
            
        Returns:
            带有标注的图像和检测到的物体列表 | Annotated image and list of detected objects
        """

        if self.pub_result_image or self.verbose:
            image = orig_img.copy()  # 复制原始图像以进行标注 | Copy original image for annotation
        else:
            image = orig_img
        detected_objects = []  # 检测到的物体列表 | List of detected objects
        
        # 处理每个检测结果 | Process each detection result
        if USE_ULTRALYTICS:
            det = results.boxes  # 获取边界框 | Get bounding boxes
        else:
            det = results

        for box in det:
            if USE_ULTRALYTICS:
                conf = box.conf.item()  # 置信度 | Confidence
                cls_id = int(box.cls.item())  # 类别ID | Class ID
                # 获取边界框坐标 | Get bounding box coordinates
                x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            else:
                *xyxy, conf, cls = box
                cls_id = int(cls)
                x0, y0, x1, y1 = map(int, xyxy)

            # 跳过置信度低于阈值的检测 | Skip detections with confidence below threshold
            if conf < self.conf_thresh:
                continue

            if cls_id < 1:
                continue
            
            # 创建物体信息字典 | Create object information dictionary
            obj_info = {
                'class': self.class_names[cls_id],  # 类别名称 | Class name
                'confidence': conf,  # 置信度 | Confidence
                'x': int((x0 + x1) / 2),  # 边界框中心x坐标 | Bounding box center x coordinate
                'y': int((y0 + y1) / 2),  # 边界框中心y坐标 | Bounding box center y coordinate
                'w': int(x1 - x0),  # 边界框宽度 | Bounding box width
                'h': int(y1 - y0)   # 边界框高度 | Bounding box height
            }
            detected_objects.append(obj_info)
            # 在图像上绘制边界框 | Draw bounding box on image
            if self.pub_result_image or self.verbose:
                color = COLORS[cls_id]
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                
        return image, detected_objects
        
    def publish_detections(self, objects):
        """
        发布检测结果到ROS话题
        Publish detection results to ROS topic
        
        Args:
            objects: 检测到的物体列表 | List of detected objects
        """
        # 创建检测结果消息 | Create detection results message
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 为每个检测到的物体创建一条消息 | Create a message for each detected object
        for obj in objects:
            detection = Detection2D()
            
            # 创建边界框消息 | Create bounding box message
            bbox = BoundingBox2D()
            
            # ROS HUMBLE格式 | ROS HUMBLE format
            bbox.center.position.x = float(obj['x'])
            bbox.center.position.y = float(obj['y'])
            bbox.center.theta = 0.0  # 旋转角度（这里设为0） | Rotation angle (set to 0 here)

            bbox.size_x = float(obj['w'])
            bbox.size_y = float(obj['h'])
            detection.bbox = bbox

            # 创建物体假设消息（包含姿态） | Create object hypothesis message (with pose)
            hypothesis = ObjectHypothesisWithPose()

            hypothesis.hypothesis.class_id = str(obj['class'])  # 类别ID | Class ID
            hypothesis.hypothesis.score = float(obj['confidence'])  # 置信度 | Confidence
            # 设置物体在相机坐标系中的位置 | Set object position in camera frame
            hypothesis.pose.pose.position = Point(x=obj['X'], y=obj['Y'], z=obj['Z'])

            detection.results.append(hypothesis)
            
            msg.detections.append(detection)
            
        # 发布检测结果消息 | Publish detection results message
        self.detection_pub.publish(msg)

def main(args):
    """
    主函数，初始化ROS节点并运行
    Main function, initialize ROS node and run
    
    Args:
        verbose: 是否输出详细信息 | Whether to output detailed information
    """
    rclpy.init()  # 初始化ROS | Initialize ROS
    node = YOLODetectorNode(print_log=args.print_log, pub_res_img=args.pub_res_img, verbose=args.verbose)  # 创建节点 | Create node

    try:
        rclpy.spin(node)  # 运行节点 | Run node
    except KeyboardInterrupt:
        pass
    finally:
        if node.verbose:
            cv2.destroyWindow(node.window_name)  # 销毁窗口 | Destroy window
        node.destroy_node()  # 销毁节点 | Destroy node
        rclpy.shutdown()  # 关闭ROS | Shutdown ROS

if __name__ == '__main__':
    # 解析命令行参数 | Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Object Detection Node')
    parser.add_argument('--print-log', action='store_true', help='Hide info')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--pub-res-img', action='store_true', help='verbose')
    args = parser.parse_args()

    main(args)  # 运行主函数 | Run main function