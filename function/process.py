import os
import glob
import cv2  # 导入OpenCV库用于处理视频
from .car_position_detector import detect_car_positions
from .lane_line_detector import detect_lane_lines
from .line_cross_judger import judge_line_cross
from sklearn.metrics import precision_score, recall_score


def data_process(input_type, input_path):
    # 根据输入类型选择处理方式
    if input_type == 'image':
        process_image(input_path)
    elif input_type == 'video':
        process_video(input_path)
    elif input_type == 'folder':
        process_folder(input_path)
    else:
        print("Unsupported input type.")


def process_image(image_path):
    # 检测汽车位置
    car_positions = detect_car_positions(image_path)

    # 检测车道线
    lane_lines = detect_lane_lines(image_path)

    # 判断是否压线
    result = judge_line_cross(car_positions, lane_lines)
    print(f"Line crossing detection result for image: {result}")


def process_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # 获取视频的帧率和分辨率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        if not ret:
            break  # 如果读取失败，退出循环

        frame_count += 1

        # 每2帧处理一次
        if frame_count % 1 == 0:
            print(f"Processing frame {frame_count}...")

            # 将当前帧保存为临时图片文件
            temp_image_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_image_path, frame)

            # 检测汽车位置
            car_positions = detect_car_positions(temp_image_path)

            # 检测车道线
            lane_lines = detect_lane_lines(temp_image_path)

            # 判断是否压线
            result = []
            if car_positions is not None and lane_lines is not None:
                result = judge_line_cross(car_positions, lane_lines)
            else:
                print("Warning: No car positions or lane lines detected.")

            # 删除临时图片文件
            os.remove(temp_image_path)

            # 在帧上绘制车道线
            if lane_lines is not None:
                for line in lane_lines:
                    if line is not None and len(line) > 0:  # 检查 line 是否有效
                        x1, y1, x2, y2 = line[0]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色车道线

            # 在帧上绘制车辆框和压线标签
            if car_positions is not None:
                for i, car in enumerate(car_positions):
                    if len(car) == 4:  # 确保 car 是 (x, y, w, h) 格式
                        x, y, w, h = car
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色车辆框
                        if result and i < len(result):  # 检查 result 是否有效
                            label = "Crossing" if result[i] else "Not Crossing"
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # 蓝色标签

        # 将处理后的帧写入输出视频
        out.write(frame)

        # 显示实时结果
        cv2.imshow("Video Processing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing completed. Output saved to {output_path}")


def process_folder(folder_path):
    # 获取文件夹中的所有图片
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

    true_labels = []
    predicted_labels = []

    for image_path in image_paths:
        # 提取图片名称
        image_name = os.path.basename(image_path)
        # 解析标签
        label = int(image_name.split('_')[-1].split('.')[0])
        true_labels.append(label)

        # 处理图片
        car_positions = detect_car_positions(image_path)
        lane_lines = detect_lane_lines(image_path)
        result = judge_line_cross(car_positions, lane_lines)

        # 将判断结果转换为标签（假设返回的是布尔值）
        predicted_label = 1 if result else 0
        predicted_labels.append(predicted_label)

    # 计算精确率和召回率
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == '__main__':
    # 示例调用
    #data_process('image', './01495.jpg')
     data_process('video', '../video/1842.mp4')
    # data_process('folder', 'path/to/folder')