import numpy as np
import cv2
from ultralytics import YOLO


def auto_detect_homography_points(image):
    """
    自动检测透视变换所需的四个点，基于图像特征
    
    参数:
    - image: 输入图像
    
    返回:
    - src_pts: 源图像中的四个点
    """
    h, w = image.shape[:2]
    
    # 根据图像尺寸自动计算透视变换的四个点
    # 假设车道在图像下半部分的梯形区域内
    src_pts = np.array([
        [int(w * 0.25), int(h * 0.65)],  # 左上
        [int(w * 0.75), int(h * 0.65)],  # 右上
        [int(w * 0.90), int(h * 0.95)],  # 右下
        [int(w * 0.10), int(h * 0.95)]   # 左下
    ], dtype='float32')
    
    return src_pts


def get_homography_matrix(image, auto_detect=True):
    """
    获取单应性矩阵
    
    参数:
    - image: 输入图像
    - auto_detect: 是否自动检测源点，默认为True
    
    返回:
    - homography_matrix: 单应性矩阵
    """
    h, w = image.shape[:2]
    
    if auto_detect:
        # 自动检测源点
        src_pts = auto_detect_homography_points(image)
    else:
        # 固定源点 (备用方案)
        src_pts = np.array([
            [int(w * 0.25), int(h * 0.65)],  # 左上
            [int(w * 0.75), int(h * 0.65)],  # 右上
            [int(w * 0.90), int(h * 0.95)],  # 右下
            [int(w * 0.10), int(h * 0.95)]   # 左下
        ], dtype='float32')
    
    # 目标坐标系中的四个点 (鸟瞰图视角)
    # 保持宽高比例一致，避免形变
    dst_width = w
    dst_height = h
    dst_pts = np.array([
        [0, 0],                  # 左上
        [dst_width, 0],          # 右上
        [dst_width, dst_height], # 右下
        [0, dst_height]          # 左下
    ], dtype='float32')
    
    # 计算单应性矩阵
    homography_matrix, status = cv2.findHomography(src_pts, dst_pts)
    
    return homography_matrix, src_pts, dst_pts


def apply_perspective_transform(image, points, homography_matrix):
    """
    对点集应用透视变换
    
    参数:
    - image: 原始图像，用于获取尺寸
    - points: 需要变换的点集 [(x, y, w, h), ...]
    - homography_matrix: 单应性矩阵
    
    返回:
    - transformed_points: 变换后的点集
    """
    transformed_points = []
    
    for point in points:
        x, y, w, h = point
        
        # 计算边界框的四个角点
        corners = np.array([
            [x, y],         # 左上
            [x + w, y],     # 右上
            [x + w, y + h], # 右下
            [x, y + h]      # 左下
        ], dtype='float32')
        
        # 扩展为齐次坐标
        corners = corners.reshape(-1, 1, 2)
        
        # 应用透视变换
        transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)
        
        # 计算变换后边界框的左上角和宽高
        min_x = np.min(transformed_corners[:, 0, 0])
        min_y = np.min(transformed_corners[:, 0, 1])
        max_x = np.max(transformed_corners[:, 0, 0])
        max_y = np.max(transformed_corners[:, 0, 1])
        
        trans_x = int(min_x)
        trans_y = int(min_y)
        trans_w = int(max_x - min_x)
        trans_h = int(max_y - min_y)
        
        transformed_points.append((trans_x, trans_y, trans_w, trans_h))
    
    return transformed_points


def detect_car_positions(image_path, apply_transform=True, visualize=True):
    """
    检测图像中的车辆位置
    
    参数:
    - image_path: 图像路径
    - apply_transform: 是否应用透视变换，默认为True
    - visualize: 是否可视化结果，默认为True
    
    返回:
    - bbox_positions: 车辆位置列表 [(x, y, w, h), ...]
    """
    # 加载YOLOv8模型
    model = YOLO("./yolov8n.pt")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法找到或读取图像文件: {image_path}")

    # 使用YOLO检测图像中的汽车
    results = model(img)

    # 获取类别名称到索引的映射
    class_names = model.names
    car_class_ids = [class_id for class_id, name in class_names.items() if name.lower() == "car"]
    if not car_class_ids:
        raise ValueError("模型中未找到类别名称为 'car' 的类别。请检查类别名称和模型配置。")
    car_class_id = car_class_ids[0]

    # 获取检测到的汽车框
    car_bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy())
            if cls_id == car_class_id:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                car_bboxes.append([x1, y1, x2, y2])
                
                # 在原始图像上标记车辆
                if visualize:
                    confidence = float(box.conf.cpu().numpy())
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Car: {confidence:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 如果启用可视化且检测到车辆，保存标记后的图像
    if visualize and car_bboxes:
        detection_output = f"detection_{image_path.split('/')[-1]}"
        cv2.imwrite(detection_output, img)
        print(f"已将标记车辆的图像保存至: {detection_output}")

    if not car_bboxes:
        print("未检测到任何汽车。")
        return []

    # 转换为 (x, y, w, h) 格式
    bbox_positions = []
    for bbox in car_bboxes:
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        bbox_positions.append((x, y, w, h))
    
    # 应用透视变换
    if apply_transform and bbox_positions:
        try:
            # 计算透视变换矩阵
            homography_matrix, src_pts, dst_pts = get_homography_matrix(img)
            
            # 可视化源点和目标透视变换区域 (调试用)
            if visualize:
                debug_img = img.copy()
                for pt in src_pts:
                    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                cv2.polylines(debug_img, [src_pts.astype(int)], True, (0, 255, 0), 2)
                
                # 保存调试图像
                cv2.imwrite("debug_perspective.jpg", debug_img)
            
            # 应用透视变换到检测到的车辆位置
            transformed_positions = apply_perspective_transform(img, bbox_positions, homography_matrix)
            
            # 如果需要，可视化变换后的结果
            if visualize:
                # 创建空白图像
                transformed_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                # 绘制变换后的车辆位置
                for pos in transformed_positions:
                    x, y, w, h = pos
                    cv2.rectangle(transformed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # 保存变换后的图像
                cv2.imwrite("transformed_view.jpg", transformed_img)
                print("已将透视变换后的车辆位置保存至: transformed_view.jpg")
                
            return transformed_positions
        except Exception as e:
            print(f"透视变换过程中出错: {e}")
            print("返回原始车辆位置...")
    
    return bbox_positions


def visualize_result(image_path, car_positions, output_path=None):
    """
    可视化检测结果
    
    参数:
    - image_path: 图像路径
    - car_positions: 车辆位置列表 [(x, y, w, h), ...]
    - output_path: 输出图像路径，默认为None (自动生成)
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 绘制车辆位置
    for i, pos in enumerate(car_positions):
        x, y, w, h = pos
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Car {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 生成输出路径
    if output_path is None:
        output_path = f"result_{image_path.split('/')[-1]}"
    
    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"结果已保存至: {output_path}")
    
    # 显示图像（可选，取决于运行环境）
    try:
        cv2.imshow("Car Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像窗口，但结果已保存到文件。")


def detect_and_visualize(image_path):
    """
    检测并可视化图像中的所有车辆
    
    参数:
    - image_path: 图像路径
    """
    try:
        # 检测车辆位置 (启用可视化)
        car_positions = detect_car_positions(image_path, apply_transform=False, visualize=True)
        
        if car_positions:
            print(f"检测到 {len(car_positions)} 辆车:")
            for i, pos in enumerate(car_positions):
                print(f"  车辆 {i+1}: {pos}")
                
            # 可视化结果
            visualize_result(image_path, car_positions)
        else:
            print("未检测到任何汽车。")
            
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    image_path = "00030.jpg"  # 替换为您的图像路径
    detect_and_visualize(image_path)
