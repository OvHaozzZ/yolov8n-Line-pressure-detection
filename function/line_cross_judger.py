import math
import numpy as np
import cv2


def point_to_line_distance(point, k, b):
    """
    计算点到直线的距离
    :param point: (x, y) 点的坐标
    :param k: 直线的斜率，如果是垂直线，则为 None
    :param b: 直线的截距，如果是垂直线，则为 x 截距
    :return: 点到直线的距离
    """
    x, y = point
    if k is None:
        # 处理垂直线的情况，b 是 x 截距
        return abs(x - b) if b is not None else float('inf')

    numerator = abs(k * x - y + b)
    denominator = math.sqrt(k ** 2 + 1)
    return numerator / denominator


def get_perspective_transform_matrix(img_width, img_height):
    """
    获取透视变换矩阵和逆矩阵
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: (M, Minv) 透视变换矩阵和逆矩阵
    """
    # 定义源点（原始图像中的四边形顶点）
    src = np.float32([
        [img_width * 0.3, img_height * 0.65],  # 左上
        [img_width * 0.7, img_height * 0.65],  # 右上
        [img_width * 0.9, img_height * 0.9],  # 右下
        [img_width * 0.1, img_height * 0.9]  # 左下
    ])

    # 定义目标点（变换后的矩形顶点）
    dst = np.float32([
        [img_width * 0.25, 0],  # 左上
        [img_width * 0.75, 0],  # 右上
        [img_width * 0.75, img_height],  # 右下
        [img_width * 0.25, img_height]  # 左下
    ])

    # 计算透视变换矩阵和逆矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def transform_points(points, M):
    """
    使用透视变换矩阵变换点集
    :param points: 点集 [(x1,y1), (x2,y2), ...]
    :param M: 透视变换矩阵
    :return: 变换后的点集
    """
    if len(points) == 0:
        return []

    # 转换为齐次坐标
    points_homogeneous = np.array([(x, y, 1) for x, y in points], dtype=np.float32).reshape(-1, 3)

    # 应用变换
    transformed_points = np.dot(M, points_homogeneous.T).T

    # 转换为笛卡尔坐标
    transformed_points_cartesian = [(x / w, y / w) for x, y, w in transformed_points]

    return transformed_points_cartesian


def transform_line_equation(k, b, M):
    """
    变换直线方程到新的坐标系
    :param k: 斜率
    :param b: 截距
    :param M: 透视变换矩阵
    :return: (new_k, new_b) 变换后的直线参数
    """
    if k is None:  # 垂直线
        # 取直线上的两个点 (b,0) 和 (b,1)
        p1 = (b, 0)
        p2 = (b, 1)
    else:
        # 取直线上的两个点 (0,b) 和 (1,k+b)
        p1 = (0, b)
        p2 = (1, k + b)

    # 变换这两个点
    transformed_p1, transformed_p2 = transform_points([p1, p2], M)

    # 计算新的直线方程
    if transformed_p1[0] == transformed_p2[0]:  # 新的垂直线
        new_k = None
        new_b = transformed_p1[0]
    else:
        new_k = (transformed_p2[1] - transformed_p1[1]) / (transformed_p2[0] - transformed_p1[0])
        new_b = transformed_p1[1] - new_k * transformed_p1[0]

    return new_k, new_b


def judge_line_cross(car_positions, lane_lines, threshold=100, img_size=(1280, 720)):
    """
    判断每辆车是否压线（加入透视变换到鸟瞰图）

    参数:
    - car_positions: list, 车辆位置列表，格式为[(x, y, w, h), ...]
    - lane_lines: list, 车道线参数列表，格式为 [k1, b1, k2, b2]
    - threshold: float, 判断压线的距离阈值（鸟瞰图中的像素距离）
    - img_size: tuple, 图像尺寸 (width, height)

    返回:
    - results: list, 每辆车是否压线的布尔值列表
    """
    if not car_positions or len(lane_lines) < 4:
        return [False] * len(car_positions)

    # 获取透视变换矩阵
    M, _ = get_perspective_transform_matrix(img_size[0], img_size[1])

    # 变换车道线方程到鸟瞰图
    transformed_lane_lines = []
    if lane_lines[0] is not None and lane_lines[1] is not None:
        k, b = transform_line_equation(lane_lines[0], lane_lines[1], M)
        transformed_lane_lines.append((k, b))  # 左车道线
    if lane_lines[2] is not None and lane_lines[3] is not None:
        k, b = transform_line_equation(lane_lines[2], lane_lines[3], M)
        transformed_lane_lines.append((k, b))  # 右车道线

    if not transformed_lane_lines:
        return [False] * len(car_positions)

    results = []
    for car in car_positions:
        is_crossing = False

        # 处理(x, y, w, h)格式的车辆位置，提取关键点
        try:
            if len(car) == 4:  # (x, y, w, h)格式
                x, y, w, h = car
                # 提取车辆的底部中心点和四个角点
                points = [
                    (x + w / 2, y + h),  # 底部中心点（最可能压线的位置）
                    (x, y + h),  # 左下角
                    (x + w, y + h)  # 右下角
                ]
            elif isinstance(car, tuple) and len(car) == 2:  # (x, y)格式
                points = [car]
            else:
                print(f"Unsupported car position format: {car}")
                points = []
        except Exception as e:
            print(f"Error processing car position: {e}")
            points = []

        # 变换车辆关键点到鸟瞰图
        transformed_points = transform_points(points, M)
        if not transformed_points:
            results.append(False)
            continue

        # 计算每个点到每条车道线的距离，并取最小值
        min_distance = float('inf')
        for point in transformed_points:
            for line in transformed_lane_lines:
                k, b = line
                try:
                    distance = point_to_line_distance(point, k, b)
                    if distance < min_distance:
                        min_distance = distance
                except Exception as e:
                    print(f"Error calculating distance: {e}, point: {point}, line: {line}")
                    continue

        # 检查最小距离是否小于阈值
        if min_distance < threshold:
            is_crossing = True

        results.append(is_crossing)

    return results


# 测试数据
car_positions_test = [(100, 100, 50, 30), (200, 200, 60, 40)]
lane_lines_test = [-0.28287794762693436, 654.3808870867997, 0.2708271639429997, 133.61501228041743]

# 测试 judge_line_cross 函数
result = judge_line_cross(car_positions_test, lane_lines_test, img_size=(1280, 720))
print(f"Test result: {result}")