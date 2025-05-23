# 基于 YOLOv8 的智能压线检测系统

<div align="center">
  <img src="docs/images/banner.jpg" alt="系统横幅" width="800"/>
  <p>基于国产大算力芯片的危险驾驶行为检测系统</p>
</div>

## 📋 目录

- [项目简介](#-项目简介)
- [系统功能](#-系统功能)
- [安装指南](#-安装指南)
- [使用方法](#-使用方法)
- [核心模块](#-核心模块)
- [技术细节](#-技术细节)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

## 🔍 项目简介

基于 YOLOv8 的智能压线检测系统是一款专为自动驾驶和智能交通领域设计的软件。该系统利用先进的 YOLOv8 目标检测算法，结合国产大算力芯片地平线 X3 派，实现对车道线的高精度识别和实时监测。通过分析行车记录仪影像资料和相关环境条件，系统能够准确判断车辆是否压线，为自动驾驶提供安全保障，并支持违规和罚款记录的查询。

本系统结合深度学习和计算机视觉技术，具有以下特点：

- **高精度检测**：采用YOLOv8目标检测算法，车辆检测准确率高达94.5%
- **实时性能**：优化的算法设计，支持实时视频分析
- **自适应能力**：适应不同天气、光照和道路条件
- **模块化架构**：便于扩展和维护的系统设计

## 💡 系统功能

### 1. 图片智能检测
- 支持 JPG、JPEG、PNG 等常见图片格式
- 自动检测图像中的车辆和车道线
- 准确判断车辆是否压线
- 生成可视化分析结果

### 2. 视频行为分析
- 支持 MP4、AVI 等常见视频格式
- 逐帧分析视频中的车辆行为
- 动态跟踪车道线变化
- 输出连续的压线判断结果

### 3. 实时安全监测 (规划中)
- 支持实时视频流输入
- 低延迟分析与预警
- 硬件加速支持
- 适配国产大算力芯片

## 📥 安装指南

### 环境要求
- Python 3.8 或更高版本
- CUDA 11.0+ (GPU加速，可选)
- OpenCV 4.5+
- PyTorch 2.0+

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/lane-crossing-detection.git
   cd lane-crossing-detection
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **下载预训练模型**
   ```bash
   # 自动下载YOLOv8n模型
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

## 🚀 使用方法

### 图片检测

```bash
# 检测单张图片
python detect_image.py --image path/to/your/image.jpg
```

### 视频分析

```bash
# 分析视频
python detect_video.py --video path/to/your/video.mp4
```

### 图形界面

```bash
# 启动图形界面
python interaction.py
```

## 🧩 核心模块

系统由以下核心模块组成：

1. **车辆检测模块** (car_position_detector.py)
   - 基于YOLOv8n模型
   - 支持透视变换
   - 输出车辆位置信息

2. **车道线检测模块** (lane_line_detector.py)
   - 基于Canny边缘检测和Hough变换
   - 自适应区域选择
   - 车道线数学拟合

3. **压线判断模块** (line_cross_judger.py)
   - 点到线距离计算
   - 多点多线检测
   - 阈值自适应调整

4. **交互界面模块** (interaction.py)
   - 基于Tkinter的现代化GUI
   - 文件选择和结果展示
   - 多种操作模式

## 🔧 技术细节

### 算法流程
1. **图像预处理**: 
   - 尺寸调整
   - 噪声过滤
   - 区域裁剪

2. **车辆检测**: 
   - YOLOv8目标检测
   - 边界框提取
   - 位置坐标转换

3. **车道线检测**: 
   - 灰度化处理
   - Canny边缘检测
   - Hough变换直线检测
   - 多项式拟合

4. **压线判断**: 
   - 计算关键点到车道线距离
   - 阈值比较
   - 结果输出

### 性能优化
- 使用轻量级YOLOv8n模型
- 图像大小与精度的平衡
- 多线程处理提高效率
- GPU加速支持

## ❓ 常见问题

### 1. 系统对图像分辨率有什么要求？
- 推荐分辨率: 640×480或更高
- 支持自动缩放，但极低分辨率会影响准确性

### 2. 检测效果受光线条件影响大吗？
- 系统对正常白天光线条件表现最佳
- 弱光和夜间场景需要预处理增强

### 3. 如何提高检测准确率？
- 使用更高分辨率的图像
- 调整检测阈值参数
- 在相似场景下微调模型

### 4. 实时检测的最低硬件要求是什么？
- CPU: Intel i5 8代或同等性能
- RAM: 8GB以上
- GPU: NVIDIA GTX 1050或更高(可选)

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <p>© 2023 行稳致远团队 版权所有</p>
  <p>
    <a href="https://github.com/yourusername/lane-crossing-detection">GitHub</a> ·
    <a href="mailto:your.email@example.com">联系我们</a>
  </p>
</div>
