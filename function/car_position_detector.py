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

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 使用HoughLinesP检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is None:
        # 如果没有检测到直线，使用默认值
        return np.array([
            [int(w * 0.25), int(h * 0.65)],  # 左上
            [int(w * 0.75), int(h * 0.65)],  # 右上
            [int(w * 0.90), int(h * 0.95)],  # 右下
            [int(w * 0.10), int(h * 0.95)]   # 左下
        ], dtype='float32')

    # 分离左右车道线
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算斜率
        if x2 != x1:  # 避免除零错误
            slope = (y2 - y1) / (x2 - x1)
            # 根据斜率判断是左车道线还是右车道线
            if slope < -0.2:  # 左车道线
                left_lines.append(line[0])
            elif slope > 0.2:  # 右车道线
                right_lines.append(line[0])

    # 如果没有检测到足够的车道线，使用默认值
    if len(left_lines) < 2 or len(right_lines) < 2:
        return np.array([
            [int(w * 0.25), int(h * 0.65)],  # 左上
            [int(w * 0.75), int(h * 0.65)],  # 右上
            [int(w * 0.90), int(h * 0.95)],  # 右下
            [int(w * 0.10), int(h * 0.95)]   # 左下
        ], dtype='float32')

    # 对左右车道线进行拟合
    left_points = np.array(left_lines).reshape(-1, 2)
    right_points = np.array(right_lines).reshape(-1, 2)

    # 使用最小二乘法拟合直线
    left_fit = np.polyfit(left_points[:, 0], left_points[:, 1], 1)
    right_fit = np.polyfit(right_points[:, 0], right_points[:, 1], 1)

    # 计算透视变换的四个点
    y_top = int(h * 0.65)  # 透视变换的上边界
    y_bottom = h  # 透视变换的下边界

    # 计算左右车道线在上下边界的x坐标
    x_left_top = int((y_top - left_fit[1]) / left_fit[0])
    x_left_bottom = int((y_bottom - left_fit[1]) / left_fit[0])
    x_right_top = int((y_top - right_fit[1]) / right_fit[0])
    x_right_bottom = int((y_bottom - right_fit[1]) / right_fit[0])

    # 确保坐标在图像范围内
    x_left_top = max(0, min(w, x_left_top))
    x_left_bottom = max(0, min(w, x_left_bottom))
    x_right_top = max(0, min(w, x_right_top))
    x_right_bottom = max(0, min(w, x_right_bottom))

    src_pts = np.array([
        [x_left_top, y_top],      # 左上
        [x_right_top, y_top],     # 右上
        [x_right_bottom, y_bottom], # 右下
        [x_left_bottom, y_bottom]  # 左下
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
            [int(w * 0.10), int(h * 0.95)]  # 左下
        ], dtype='float32')

    # 目标坐标系中的四个点 (鸟瞰图视角)
    # 保持宽高比例一致，避免形变
    dst_width = w
    dst_height = h
    dst_pts = np.array([
        [0, 0],  # 左上
        [dst_width, 0],  # 右上
        [dst_width, dst_height],  # 右下
        [0, dst_height]  # 左下
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
            [x, y],  # 左上
            [x + w, y],  # 右上
            [x + w, y + h],  # 右下
            [x, y + h]  # 左下
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


def detect_car_positions(image_path, apply_transform=True, visualize=False):
    """
    检测图像中的车辆位置

    参数:
    - image_path: 图像路径
    - apply_transform: 是否应用透视变换，默认为True
    - visualize: 是否可视化结果，默认为False

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

            # 应用透视变换到检测到的车辆位置
            transformed_positions = apply_perspective_transform(img, bbox_positions, homography_matrix)

            return transformed_positions
        except Exception as e:
            print(f"透视变换过程中出错: {e}")
            print("返回原始车辆位置...")

    return bbox_positions


if __name__ == "__main__":
    image_path = "../photo/03780.jpg"  # 替换为您的图像路径
    car_positions = detect_car_positions(image_path, apply_transform=False, visualize=False)
    if car_positions:
        print(f"检测到 {len(car_positions)} 辆车:")
        for i, pos in enumerate(car_positions):
            print(f"  车辆 {i + 1}: {pos}")
    else:
        print("未检测到任何汽车。")