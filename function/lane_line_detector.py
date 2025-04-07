#!/usr/bin/python
# -*- encoding: utf-8 -*-

# Built-in modules
import argparse

# Third-party modules
import cv2
import numpy as np


class LaneDetection:
    """车道线检测"""

    def __init__(
            self,
            ksize=(5, 5),
            sigma=(0, 0),
            threshold1=100,
            threshold2=200,
            aperture_size=3,
            direction_point=None,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            min_line_len=200,
            max_line_gap=400,
            x1L=None,
            x2L=None,
            x1R=None,
            x2R=None,
    ):
        self.ksize = ksize
        self.sigma = sigma
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.direction_point = direction_point
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.x1L = x1L
        self.x2L = x2L
        self.x1R = x1R
        self.x2R = x2R

    def __call__(self, img):
        gauss = self._image_preprocess(img)
        edge = self._edge_canny(gauss)
        roi = self._roi_trapezoid(edge)
        lines = self._Hough_line_fitting(roi)
        p_list = self._lane_line_fitting(img, lines)
        return p_list

    def _image_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, self.ksize, self.sigma[0], self.sigma[1])
        return gauss

    def _edge_canny(self, img):
        edge = cv2.Canny(img, self.threshold1, self.threshold2, self.aperture_size)
        return edge

    def _roi_trapezoid(self, img):
        h, w = img.shape[:2]
        top_of_interest = int(h * 0.2)  # 只考虑图像的上五分之四部分

        if self.direction_point is None:
            left_top = [w // 2, top_of_interest]
            right_top = [w // 2, top_of_interest]
        else:
            left_top = self.direction_point
            right_top = self.direction_point

        left_down = [int(w * 0.1), h]
        right_down = [int(w * 0.9), h]
        self.roi_points = np.array([left_down, left_top, right_top, right_down])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.roi_points, 255)
        roi = cv2.bitwise_and(img, mask)
        return roi

    def _Hough_line_fitting(self, img):
        lines = cv2.HoughLinesP(
            img, self.rho, self.theta, self.threshold, np.array([]),
            minLineLength=self.min_line_len, maxLineGap=self.max_line_gap
        )
        if lines is None:
            print("Warning: No lines detected by HoughLinesP.")
            return []  # 返回空列表，避免后续迭代报错
        return lines

    def _lane_line_fitting(self, img, lines):
        if lines is None or len(lines) == 0:
            print("Warning: No lines to fit.")
            return [None, None, None, None]  # 返回空列表，避免后续处理报错

        right_x = []
        right_y = []
        left_x = []
        left_y = []

        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # 避免除零错误
                    if x2 == x1:
                        continue

                    slope = ((y2 - y1) / (x2 - x1))
                    if slope <= -0.2:
                        left_x.extend((x1, x2))
                        left_y.extend((y1, y2))
                    elif slope >= 0.2:
                        right_x.extend((x1, x2))
                        right_y.extend((y1, y2))
        except Exception as e:
            print(f"Error in line processing: {e}")

        left_fit = None
        right_fit = None

        # 处理左车道线
        if left_x and left_y:
            try:
                left_fit = np.polyfit(left_x, left_y, 1)
            except Exception as e:
                print(f"Error fitting left lane line: {e}")
                left_fit = None

        # 处理右车道线
        if right_x and right_y:
            try:
                right_fit = np.polyfit(right_x, right_y, 1)
            except Exception as e:
                print(f"Error fitting right lane line: {e}")
                right_fit = None

        # 创建结果列表
        k_left = left_fit[0] if left_fit is not None else None
        b_left = left_fit[1] if left_fit is not None else None
        k_right = right_fit[0] if right_fit is not None else None
        b_right = right_fit[1] if right_fit is not None else None

        return [k_left, b_left, k_right, b_right]


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection V1.0")
    parser.add_argument("-i", "--input_path", type=str, default="./assets/1.jpg", help="Input path of image.")
    parser.add_argument("-o", "--output_path", type=str, default="./assets/1_out.jpg", help="Output path of image.")
    return parser.parse_args()


def detect_lane_lines(input_path="./00030.jpg"):
    try:
        args = parse_args()  # 输入输出的路径

        lanedetection = LaneDetection()

        # jpg图片检测
        if input_path.endswith('.jpg'):
            img = cv2.imread(input_path, 1)
            if img is None:
                print(f"Error: Unable to read image at {input_path}")
                return [None, None, None, None]

            p_list = lanedetection(img)
            if p_list:
                print(f"Left lane line equation: y = {p_list[0]}x + {p_list[1]}")
                print(f"Right lane line equation: y = {p_list[2]}x + {p_list[3]}")
            else:
                print("Failed to detect lane lines.")
                p_list = [None, None, None, None]
            return p_list

        # mp4视频检测
        elif input_path.endswith('.mp4'):
            video_capture = cv2.VideoCapture(input_path)
            if not video_capture.isOpened():
                print('Open is false!')
                return [None, None, None, None]

            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("fps: {} \nsize: {}".format(fps, size))

            total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print("[INFO] {} total frames in video".format(total))

            frameToStart = 0
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

            p_list = [None, None, None, None]
            try:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    try:
                        p_list = lanedetection(frame)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue

                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Error in video loop: {e}")
            finally:
                video_capture.release()
                cv2.destroyAllWindows()

            if p_list[0] is not None and p_list[1] is not None:
                print(f"Left lane line equation: y = {p_list[0]}x + {p_list[1]}")
            if p_list[2] is not None and p_list[3] is not None:
                print(f"Right lane line equation: y = {p_list[2]}x + {p_list[3]}")

            return p_list

        else:
            print(f"Unsupported file format: {input_path}")
            return [None, None, None, None]

    except Exception as e:
        print(f"Error in detect_lane_lines: {e}")
        return [None, None, None, None]

    cv2.waitKey(0)


if __name__ == "__main__":
    try:
        p_list = detect_lane_lines(input_path="02610.jpg")

        print(p_list)
    except Exception as e:
        print(f"Error in main: {e}")



