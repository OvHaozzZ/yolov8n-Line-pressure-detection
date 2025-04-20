import sys
import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QLinearGradient, QGradient, QPainter, QImage
from PyQt6.QtCore import pyqtProperty, QParallelAnimationGroup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 导入数据处理模块
try:
    from function.process import data_process
except ImportError:
    # 如果模块不存在，创建一个占位符函数
    def data_process(mode, path):
        logging.warning(f"数据处理模块未找到，模式: {mode}, 路径: {path}")
        return None

class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(250, 60)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # 创建悬浮动画
        self._hover_animation = QPropertyAnimation(self, b"geometry")
        self._hover_animation.setDuration(200)
        
        # 创建点击动画
        self._click_animation = QPropertyAnimation(self, b"geometry")
        self._click_animation.setDuration(100)
        
    def enterEvent(self, event):
        # 鼠标悬浮时向上移动
        geo = self.geometry()
        self._hover_animation.setStartValue(geo)
        self._hover_animation.setEndValue(QRect(geo.x(), geo.y() - 5, geo.width(), geo.height()))
        self._hover_animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        # 鼠标离开时恢复位置
        geo = self.geometry()
        self._hover_animation.setStartValue(geo)
        self._hover_animation.setEndValue(QRect(geo.x(), geo.y() + 5, geo.width(), geo.height()))
        self._hover_animation.start()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        # 点击时缩小
        geo = self.geometry()
        self._click_animation.setStartValue(geo)
        self._click_animation.setEndValue(QRect(
            geo.x() + 2, geo.y() + 2, 
            geo.width() - 4, geo.height() - 4
        ))
        self._click_animation.start()
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        # 释放时恢复大小
        geo = self.geometry()
        self._click_animation.setStartValue(geo)
        self._click_animation.setEndValue(QRect(
            geo.x() - 2, geo.y() - 2,
            geo.width() + 4, geo.height() + 4
        ))
        self._click_animation.start()
        super().mouseReleaseEvent(event)

class RoadLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(4)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 0.6);")
        
        # 创建移动动画
        self._animation = QPropertyAnimation(self, b"pos")
        self._animation.setDuration(3000)  # 3秒
        self._animation.setLoopCount(-1)   # 无限循环
        
    def startAnimation(self, start_x, end_x, y, delay=0):
        self._animation.setStartValue(QPoint(start_x, y))
        self._animation.setEndValue(QPoint(end_x, y))
        self._animation.setEasingCurve(QEasingCurve.Type.Linear)
        self._animation.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行稳致远 - 压线检测系统")
        self.setMinimumSize(1000, 600)
        
        # 设置背景
        self.background_images = [
            os.path.join("images", f"background{i}.jpg") for i in range(1, 5)
        ]
        self.current_bg_index = 0
        self.current_bg = QImage(self.background_images[self.current_bg_index])
        if self.current_bg.isNull():
            logging.error(f"Failed to load background image: {self.background_images[self.current_bg_index]}")
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建标题
        title_label = QLabel("行稳致远")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 48px;
                font-weight: bold;
            }
        """)
        
        # 创建副标题
        subtitle_label = QLabel("基于国产大算力芯片的压线检测系统")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 18px;
                margin-bottom: 20px;
            }
        """)
        
        # 创建按钮
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        self.image_btn = AnimatedButton("📷 图片智能检测")
        self.video_btn = AnimatedButton("🎥 视频行为分析")
        self.monitor_btn = AnimatedButton("🚨 实时安全监测")
        
        button_layout.addWidget(self.image_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.video_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.monitor_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 添加到主布局
        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # 创建页脚
        footer = QLabel("© 2025 行稳致远系统 版权所有")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("""
            QLabel {
                color: #cccccc;
                padding: 10px;
                background-color: rgba(0, 0, 0, 0.5);
            }
        """)
        layout.addWidget(footer)
        
        # 连接信号
        self.image_btn.clicked.connect(self.open_image)
        self.video_btn.clicked.connect(self.open_video)
        self.monitor_btn.clicked.connect(self.show_monitor_warning)
        
        # 创建背景切换定时器
        self.bg_timer = QTimer(self)
        self.bg_timer.timeout.connect(self.change_background)
        self.bg_timer.start(15000)  # 每15秒切换一次背景
        
        # 创建车道线
        self.create_road_lines()
        
        # 设置初始背景
        self.update()
        
    def create_road_lines(self):
        """创建动态车道线"""
        for i in range(5):
            line = RoadLine(self)
            line.setFixedWidth(100 + i * 20)  # 不同宽度
            y_pos = 300 + i * 50  # 不同垂直位置
            line.startAnimation(-line.width(), self.width(), y_pos, i * 500)
            
    def paintEvent(self, event):
        """绘制背景"""
        painter = QPainter(self)
        scaled_bg = self.current_bg.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        
        # 计算居中位置
        x = (scaled_bg.width() - self.width()) // 2
        y = (scaled_bg.height() - self.height()) // 2
        
        # 绘制背景
        painter.drawImage(0, 0, scaled_bg, x, y, self.width(), self.height())
        
        # 添加半透明遮罩使界面元素更清晰
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
    def change_background(self):
        """切换背景图片"""
        self.current_bg_index = (self.current_bg_index + 1) % len(self.background_images)
        self.current_bg = QImage(self.background_images[self.current_bg_index])
        self.update()  # 触发重绘
        
    def open_image(self):
        """打开图片文件进行处理"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)"
            )
            if file_path:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                # 调用数据处理模块
                data_process('image', file_path)
        except Exception as e:
            logging.error(f"图片处理失败: {e}")
            QMessageBox.critical(self, "错误", f"图片处理失败: {str(e)}")
            
    def open_video(self):
        """打开视频文件进行处理"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频文件 (*.mp4 *.avi)"
            )
            if file_path:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                # 调用数据处理模块
                data_process('video', file_path)
        except Exception as e:
            logging.error(f"视频处理失败: {e}")
            QMessageBox.critical(self, "错误", f"视频处理失败: {str(e)}")
            
    def show_monitor_warning(self):
        """显示监控警告"""
        try:
            QMessageBox.warning(
                self, 
                "设备状态",
                "⛔ 未检测到安全监测设备\n请连接专用硬件后重试"
            )
        except Exception as e:
            logging.error(f"实时监控功能失败: {e}")
            QMessageBox.critical(self, "错误", f"实时监控功能失败: {str(e)}")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            reply = QMessageBox.question(
                self, '退出', '确定要退出程序吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()
        except Exception as e:
            logging.error(f"关闭窗口时发生错误: {e}")
            event.accept()  # 出错时直接关闭

def main():
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 