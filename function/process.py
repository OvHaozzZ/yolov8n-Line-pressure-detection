import os
import glob
import cv2
from .car_position_detector import detect_car_positions
from .lane_line_detector import detect_lane_lines
from .line_cross_judger import judge_line_cross
from sklearn.metrics import precision_score, recall_score
import tempfile
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
def data_process(input_type, input_path):
    if input_type == 'image':
        process_image(input_path)
    elif input_type == 'video':
        process_video(input_path)
    elif input_type == 'folder':
        process_folder(input_path)
    else:
        print("Unsupported input type.")


def adjust_coordinates(coord, image_shape):
    """Adjust coordinates to ensure they are within the image bounds."""
    height, width = image_shape[:2]
    return max(0, min(width - 1, coord))


def save_temp_image(image):
    """Save a temporary image file and return its path."""
    _, temp_file_path = tempfile.mkstemp(suffix='.jpg')
    cv2.imwrite(temp_file_path, image)
    return temp_file_path


def delete_temp_image(file_path):
    """Delete the temporary image file."""
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Failed to delete temporary file {file_path}: {e}")


def add_legend(image, legend_items):
    """Add a legend to the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 20
    start_y = 30

    for i, (text, color) in enumerate(legend_items.items()):
        # Add color box
        cv2.rectangle(image, (margin, start_y + i * 40 - 20),
                      (margin + 30, start_y + i * 40 + 10), color, -1)
        # Add text
        cv2.putText(image, text, (margin + 40, start_y + i * 40),
                    font, font_scale, (255, 255, 255), thickness)


def process_image(image_path):
    # 1. 读取原始图像
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 2. 保存临时图像
    temp_img_path = save_temp_image(img)

    # 3. 检测车道线
    k_left, b_left, k_right, b_right = detect_lane_lines(temp_img_path)

    # 4. 检测车辆
    cars = detect_car_positions(temp_img_path, apply_transform=False)

    # 5. 删除临时图像
    delete_temp_image(temp_img_path)

    # 6. 判断压线
    lane_lines = [k_left, b_left, k_right, b_right]
    results = judge_line_cross(cars, lane_lines) if cars else []

    # 7. 绘制车道线
    y_top = int(h * 0.6)
    y_bottom = h

    # 左车道线
    if k_left is not None and b_left is not None:
        x1_left = int((y_top - b_left) / k_left) if k_left != 0 else 0
        x2_left = int((y_bottom - b_left) / k_left) if k_left != 0 else 0
        cv2.line(img, (x1_left, y_top), (x2_left, y_bottom), (0, 255, 255), 2)  # 黄色车道线

    # 右车道线
    if k_right is not None and b_right is not None:
        x1_right = int((y_top - b_right) / k_right) if k_right != 0 else w
        x2_right = int((y_bottom - b_right) / k_right) if k_right != 0 else w
        cv2.line(img, (x1_right, y_top), (x2_right, y_bottom), (0, 255, 255), 2)  # 黄色车道线

    # 8. 绘制车辆框和标签
    if cars:
        for i, (x, y, w, h) in enumerate(cars):
            # 根据压线结果选择颜色
            color = (0, 0, 255) if (i < len(results) and results[i]) else (0, 255, 0)
            thickness = 2

            # 绘制车辆框
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

            # 添加标签
            label = "Crossing" if (i < len(results) and results[i]) else "Safe"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

    # 9. 添加图例
    legend_items = {
        "Crossing (Red)": (0, 0, 255),
        "Not Crossing (Green)": (0, 255, 0),
        "Lane Line (Yellow)": (0, 255, 255)
    }
    add_legend(img, legend_items)

    # 10. 显示结果
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path):
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 准备输出视频
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"开始处理视频: {video_path}")
    print(f"视频信息: {frame_width}x{frame_height}, {fps}FPS, 总帧数: {total_frames}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0 or frame_count == total_frames:
            print(f"处理进度: {frame_count}/{total_frames} ({frame_count / total_frames:.1%})")

        try:
            # 1. 保存当前帧为临时图片
            temp_img_path = save_temp_image(frame)

            # 2. 使用process_image的核心处理逻辑
            # 检测车道线
            k_left, b_left, k_right, b_right = detect_lane_lines(temp_img_path)
            # 检测车辆
            cars = detect_car_positions(temp_img_path, apply_transform=False)
            # 删除临时文件
            delete_temp_image(temp_img_path)

            # 3. 判断压线
            lane_lines = [k_left, b_left, k_right, b_right]
            results = judge_line_cross(cars, lane_lines) if cars else []

            # 4. 绘制车道线
            y_top = int(frame_height * 0.6)
            y_bottom = frame_height

            if k_left is not None and b_left is not None:
                x1_left = int((y_top - b_left) / k_left) if k_left != 0 else 0
                x2_left = int((y_bottom - b_left) / k_left) if k_left != 0 else 0
                cv2.line(frame, (x1_left, y_top), (x2_left, y_bottom), (0, 255, 255), 2)

            if k_right is not None and b_right is not None:
                x1_right = int((y_top - b_right) / k_right) if k_right != 0 else frame_width
                x2_right = int((y_bottom - b_right) / k_right) if k_right != 0 else frame_width
                cv2.line(frame, (x1_right, y_top), (x2_right, y_bottom), (0, 255, 255), 2)

            # 5. 绘制车辆框和标签
            if cars:
                for i, (x, y, w, h) in enumerate(cars):
                    color = (0, 0, 255) if (i < len(results) and results[i]) else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = "Crossing" if (i < len(results) and results[i]) else "Safe"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 6. 添加图例
            legend_items = {
                "Crossing (Red)": (0, 0, 255),
                "Not Crossing (Green)": (0, 255, 0),
                "Lane Line (Yellow)": (0, 255, 255)
            }
            add_legend(frame, legend_items)

            # 7. 写入输出视频
            out.write(frame)

            # 8. 显示处理过程
            cv2.imshow("Video Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"处理第 {frame_count} 帧时出错: {e}")
            continue

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成. 结果保存到: {output_path}")



def process_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    true_labels = []
    predicted_labels = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        label = int(image_name.split('_')[-1].split('.')[0])
        true_labels.append(label)

        img = cv2.imread(image_path)
        temp_img_path = save_temp_image(img)
        car_positions = detect_car_positions(temp_img_path)
        lane_lines = detect_lane_lines(temp_img_path)
        result = judge_line_cross(car_positions, lane_lines)
        delete_temp_image(temp_img_path)

        predicted_label = 1 if any(result) else 0
        predicted_labels.append(predicted_label)

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == '__main__':
    data_process('image', './01495.jpg')
    # data_process('video', '../video/1842.mp4')
    # data_process('folder', 'path/to/folder')