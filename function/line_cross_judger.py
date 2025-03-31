import math
import numpy as np


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


def judge_line_cross(car_positions, lane_lines, threshold=10):
    """
    判断每辆车是否压线。

    参数:
    - car_positions: list, 车辆位置列表，格式为[(x, y, w, h), ...]
    - lane_lines: list, 车道线参数列表，格式为 [k1, b1, k2, b2]
    - threshold: float, 判断压线的距离阈值

    返回:
    - results: list, 每辆车是否压线的布尔值列表
    """
    if not car_positions or len(lane_lines) < 4:
        return [False] * len(car_positions)

    # 确保车道线参数有效
    lane_line_pairs = []
    if lane_lines[0] is not None and lane_lines[1] is not None:
        lane_line_pairs.append((lane_lines[0], lane_lines[1]))  # 左车道线
    if lane_lines[2] is not None and lane_lines[3] is not None:
        lane_line_pairs.append((lane_lines[2], lane_lines[3]))  # 右车道线
    
    if not lane_line_pairs:
        return [False] * len(car_positions)

    results = []
    for car in car_positions:
        is_crossing = False
        
        # 处理(x, y, w, h)格式的车辆位置，提取关键点
        try:
            if len(car) == 4:  # (x, y, w, h)格式
                x, y, w, h = car
                # 提取车辆的四个角点
                points = [
                    (x, y),           # 左上
                    (x + w, y),       # 右上
                    (x, y + h),       # 左下
                    (x + w, y + h)    # 右下
                ]
            elif isinstance(car, tuple) and len(car) == 2:  # (x, y)格式
                points = [car]
            else:
                print(f"Unsupported car position format: {car}")
                points = []
        except Exception as e:
            print(f"Error processing car position: {e}")
            points = []

        # 检查车辆的每个点是否在车道线附近
        for line in lane_line_pairs:
            k, b = line
            for point in points:
                try:
                    distance = point_to_line_distance(point, k, b)
                    if distance < threshold:
                        is_crossing = True
                        break
                except Exception as e:
                    print(f"Error calculating distance: {e}, point: {point}, line: {line}")
                    continue
            
            if is_crossing:
                break

        results.append(is_crossing)
    
    return results