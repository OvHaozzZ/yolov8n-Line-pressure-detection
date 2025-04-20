import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
import ctypes
import logging
from pathlib import Path
from typing import Optional
from function.process import data_process

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()

# 常量定义
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
TITLE_FONT_SIZE = 36
SUBTITLE_FONT_SIZE = 14
BUTTON_FONT_SIZE = 12

# 加载Windows系统字体
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    DEFAULT_FONT = ("Microsoft YaHei", BUTTON_FONT_SIZE, "bold")
except Exception as e:
    logging.warning(f"无法设置DPI感知: {e}")
    DEFAULT_FONT = ("Helvetica", BUTTON_FONT_SIZE, "bold")


class ResourceManager:
    """资源管理器类"""

    @staticmethod
    def load_image(path: str, size: tuple) -> Optional[ImageTk.PhotoImage]:
        try:
            image = Image.open(path)
            image = image.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception as e:
            logging.error(f"加载图片失败: {e}")
            return None


class LuxuryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("行稳致远")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # 设置窗口最小尺寸
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # 初始化UI组件
        self._init_ui()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _init_ui(self):
        """初始化UI组件"""
        try:
            # 加载背景图
            bg_path = SCRIPT_DIR / "background.jpg"
            if not bg_path.exists():
                raise FileNotFoundError(f"背景图片不存在: {bg_path}")

            self.bg_photo = ResourceManager.load_image(str(bg_path), (WINDOW_WIDTH, WINDOW_HEIGHT))
            if not self.bg_photo:
                raise RuntimeError("背景图片加载失败")

            # 创建主画布
            self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor=tk.NW)

            # 绘制标题
            self._draw_title()

            # 创建按钮框架
            self.button_frame = ttk.Frame(self.canvas, style="Luxury.TFrame")
            self.button_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            # 配置样式
            self._configure_styles()

            # 创建按钮
            self._create_buttons()

        except Exception as e:
            logging.error(f"UI初始化失败: {e}")
            messagebox.showerror("错误", f"程序初始化失败: {str(e)}")
            self.root.destroy()

    def _draw_title(self):
        """绘制标题"""
        try:
            # 主标题
            self.canvas.create_text(
                WINDOW_WIDTH / 2, 100,
                text="行稳致远",
                font=("Microsoft YaHei", TITLE_FONT_SIZE, "bold"),
                fill="#ffffff",
                anchor=tk.CENTER
            )

            # 副标题
            self.canvas.create_text(
                WINDOW_WIDTH / 2, 150,
                text="基于国产大算力芯片的压线检测系统",
                font=("Microsoft YaHei", SUBTITLE_FONT_SIZE),
                fill="#cccccc",
                anchor=tk.CENTER
            )

            # 装饰线
            self.canvas.create_line(
                WINDOW_WIDTH / 2 - 150, 130,
                WINDOW_WIDTH / 2 + 150, 130,
                fill="#3498db",
                width=3,
                capstyle=tk.ROUND
            )
        except Exception as e:
            logging.error(f"绘制标题失败: {e}")

    def _configure_styles(self):
        """配置样式"""
        try:
            self.style = ttk.Style()

            # 配置框架样式
            self.style.configure(
                "Luxury.TFrame",
                background="",
                borderwidth=0
            )

            # 配置按钮样式
            self.style.map(
                "Luxury.TButton",
                foreground=[('active', '#ffffff'), ('!active', '#000000')],
                background=[('active', '#2980b9'), ('!active', '#3498db')],
                bordercolor=[('active', '#2980b9')],
                lightcolor=[('active', '#3498db')],
                darkcolor=[('active', '#2980b9')]
            )

            self.style.configure(
                "Luxury.TButton",
                font=DEFAULT_FONT,
                padding=15,
                width=15,
                borderwidth=3,
                relief="raised",
                anchor=tk.CENTER
            )
        except Exception as e:
            logging.error(f"样式配置失败: {e}")

    def _create_buttons(self):
        """创建按钮"""
        try:
            buttons = [
                ("📷 图片智能检测", self.open_image),
                ("🎥 视频行为分析", self.open_video),
                ("🚨 实时安全监测", self.realtime_monitor)
            ]

            for idx, (text, command) in enumerate(buttons):
                btn = ttk.Button(
                    self.button_frame,
                    text=text,
                    command=command,
                    style="Luxury.TButton"
                )
                btn.grid(row=0, column=idx, padx=20, pady=20, ipadx=10, ipady=10)
        except Exception as e:
            logging.error(f"创建按钮失败: {e}")

    def _on_closing(self):
        """窗口关闭处理"""
        try:
            if messagebox.askokcancel("退出", "确定要退出程序吗？"):
                self.root.destroy()
        except Exception as e:
            logging.error(f"关闭窗口时发生错误: {e}")
            self.root.destroy()

    def open_image(self):
        """打开图片处理"""
        try:
            filetypes = [("图像文件", "*.jpg *.jpeg *.png")]
            path = filedialog.askopenfilename(title="选择待检测图片", filetypes=filetypes)
            if path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在: {path}")
                data_process('image', path)
        except Exception as e:
            logging.error(f"图片处理失败: {e}")
            messagebox.showerror("错误", f"图片处理失败: {str(e)}")

    def open_video(self):
        """打开视频处理"""
        try:
            filetypes = [("视频文件", "*.mp4 *.avi")]
            path = filedialog.askopenfilename(title="选择待分析视频", filetypes=filetypes)
            if path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在: {path}")
                data_process('video', path)
        except Exception as e:
            logging.error(f"视频处理失败: {e}")
            messagebox.showerror("错误", f"视频处理失败: {str(e)}")

    def realtime_monitor(self):
        """实时监控"""
        try:
            messagebox.showwarning(
                "设备状态",
                "⛔ 未检测到安全监测设备\n请连接专用硬件后重试"
            )
        except Exception as e:
            logging.error(f"实时监控失败: {e}")
            messagebox.showerror("错误", f"实时监控失败: {str(e)}")


def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = LuxuryApp(root)

        # 设置窗口图标
        icon_path = SCRIPT_DIR / "system_icon.ico"
        if icon_path.exists():
            try:
                root.iconbitmap(str(icon_path))
            except Exception as e:
                logging.warning(f"设置窗口图标失败: {e}")

        root.mainloop()
    except Exception as e:
        logging.critical(f"程序运行失败: {e}")
        messagebox.showerror("严重错误", f"程序运行失败: {str(e)}")


if __name__ == "__main__":
    main()



