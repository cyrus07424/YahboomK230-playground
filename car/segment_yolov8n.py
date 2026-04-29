from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import utime
import image
import random
import gc
import sys
import aidemo
from ybUtils.YbBuzzer import YbBuzzer

# 自定义YOLOv8分割类 Custom YOLOv8 Segmentation Class


class BuzzerAlert:
    def __init__(self, frequency=2000, volume=50, duration=0.1, interval_ms=300):
        self.buzzer = YbBuzzer()
        self.frequency = frequency
        self.volume = volume
        self.duration = duration
        self.interval_ms = interval_ms
        self.next_alert_ms = None
        self.off()

    def update(self, enabled):
        if not enabled:
            self.off()
            self.next_alert_ms = None
            return

        now_ms = utime.ticks_ms()
        if self.next_alert_ms is None or utime.ticks_diff(now_ms, self.next_alert_ms) >= 0:
            self.buzzer.on(self.frequency, self.volume, self.duration)
            self.next_alert_ms = utime.ticks_add(now_ms, self.interval_ms)

    def off(self):
        self.buzzer.off()


class SegmentationApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size, confidence_threshold=0.2, nms_threshold=0.5, mask_threshold=0.5, rgb888p_size=[224, 224], display_size=[1920, 1080], debug_mode=0, alert_labels=("person", "car", "truck")):
        """
        初始化分割应用类
        Initialize the segmentation application class

        参数：
        Parameters:
            kmodel_path: 模型文件路径 / Path to the kmodel file
            labels: 类别标签列表 / List of class labels
            model_input_size: 模型输入尺寸 / Model input size
            confidence_threshold: 置信度阈值 / Confidence threshold
            nms_threshold: 非极大值抑制阈值 / Non-maximum suppression threshold
            mask_threshold: 掩码阈值 / Mask threshold
            rgb888p_size: RGB图像尺寸 / RGB image size
            display_size: 显示尺寸 / Display size
            debug_mode: 调试模式 / Debug mode
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        # 模型路径 / Model path
        self.kmodel_path = kmodel_path
        # 分割类别标签 / Segmentation class labels
        self.labels = labels
        self.alert_label_ids = set()
        for label in alert_labels:
            if label in self.labels:
                self.alert_label_ids.add(self.labels.index(label))
        # 模型输入分辨率 / Model input resolution
        self.model_input_size = model_input_size
        # 置信度阈值 / Confidence threshold
        self.confidence_threshold = confidence_threshold
        # nms阈值 / NMS threshold
        self.nms_threshold = nms_threshold
        # mask阈值 / Mask threshold
        self.mask_threshold = mask_threshold
        # sensor给到AI的图像分辨率（宽度对齐到16的倍数）
        # Image resolution from sensor to AI (width aligned to multiple of 16)
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 显示分辨率（宽度对齐到16的倍数）
        # Display resolution (width aligned to multiple of 16)
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        # 检测框预置颜色值 (ARGB格式: Alpha, Red, Green, Blue)
        # Preset colors for detection boxes (ARGB format: Alpha, Red, Green, Blue)
        self.color_four = [(255, 220, 20, 60), (255, 119, 11, 32), (255, 0, 0, 142), (255, 0, 0, 230),
                         (255, 106, 0, 228), (255, 0, 60, 100), (255, 0, 80, 100), (255, 0, 0, 70),
                         (255, 0, 0, 192), (255, 250, 170, 30), (255, 100, 170, 30), (255, 220, 220, 0),
                         (255, 175, 116, 175), (255, 250, 0, 30), (255, 165, 42, 42), (255, 255, 77, 255),
                         (255, 0, 226, 252), (255, 182, 182, 255), (255, 0, 82, 0), (255, 120, 166, 157)]
        # 分割结果的numpy.array，用于给到aidemo后处理接口
        # Numpy array for segmentation results, used for the aidemo post-processing interface
        self.masks = np.zeros((1, self.display_size[1], self.display_size[0], 4))
        # Ai2d实例，用于实现模型预处理
        # Ai2d instance for model preprocessing
        self.ai2d = Ai2d(debug_mode)
        # 设置Ai2d的输入输出格式和类型
        # Set Ai2d input and output formats and types
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """
        配置预处理操作
        Configure preprocessing operations

        参数：
        Parameters:
            input_image_size: 输入图像尺寸，如果为None则使用默认尺寸
                            Input image size, if None, use default size
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 初始化ai2d预处理配置，默认为sensor给到AI的尺寸，您可以通过设置input_image_size自行修改输入尺寸
            # Initialize ai2d preprocessing configuration, default is the size from sensor to AI,
            # you can modify the input size by setting input_image_size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            # 配置padding操作，保持宽高比
            # Configure padding operation to maintain aspect ratio
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            # 配置resize操作，使用双线性插值
            # Configure resize operation using bilinear interpolation
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # 构建ai2d处理图
            # Build ai2d processing graph
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        """
        自定义当前任务的后处理
        Custom post-processing for the current task

        参数：
        Parameters:
            results: 模型推理结果 / Model inference results

        返回：
        Returns:
            seg_res: 分割结果 / Segmentation results
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 这里使用了aidemo的segment_postprocess接口进行后处理
            # Using aidemo's segment_postprocess interface for post-processing
            seg_res = aidemo.segment_postprocess(
                results,
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                self.model_input_size,
                [self.display_size[1], self.display_size[0]],
                self.confidence_threshold,
                self.nms_threshold,
                self.mask_threshold,
                self.masks
            )
            return seg_res

    def draw_result(self, pl, seg_res):
        """
        绘制分割结果到显示层
        Draw segmentation results to display layer

        参数：
        Parameters:
            pl: Pipeline对象 / Pipeline object
            seg_res: 分割结果 / Segmentation results
        """
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if seg_res[0]:  # 如果有检测到物体 / If objects are detected
                pl.osd_img.clear()  # 清除OSD图层 / Clear OSD layer
                # 创建引用mask数据的图像对象 / Create image object referencing mask data
                mask_img = image.Image(self.display_size[0], self.display_size[1], image.ARGB8888, alloc=image.ALLOC_REF, data=self.masks)
                pl.osd_img.copy_from(mask_img)  # 复制mask图像到OSD层 / Copy mask image to OSD layer

                # 提取检测结果 / Extract detection results
                dets, ids, scores = seg_res[0], seg_res[1], seg_res[2]
                for i, det in enumerate(dets):
                    # 绘制标签和置信度 / Draw label and confidence
                    x1, y1, w, h = map(lambda x: int(round(x, 0)), det)
                    pl.osd_img.draw_string_advanced(
                        x1, y1-50, 32,
                        " " + self.labels[int(ids[i])] + " " + str(round(scores[i], 2)),
                        color=self.get_color(int(ids[i]))
                    )
            else:
                pl.osd_img.clear()  # 没有检测结果时清除OSD / Clear OSD when no detection

    def has_alert_target(self, seg_res):
        if not seg_res[0]:
            return False

        ids = seg_res[1]
        for class_id in ids:
            if int(class_id) in self.alert_label_ids:
                return True
        return False

    def get_padding_param(self):
        """
        计算保持宽高比的padding参数
        Calculate padding parameters to maintain aspect ratio

        返回：
        Returns:
            top, bottom, left, right: padding参数 / padding parameters
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        # 计算缩放比例 / Calculate scaling ratio
        ratio_w = float(dst_w) / self.rgb888p_size[0]
        ratio_h = float(dst_h) / self.rgb888p_size[1]
        # 选择较小的比例，以确保整个图像都能放入 / Choose smaller ratio to ensure entire image fits
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        # 计算缩放后的新尺寸 / Calculate new dimensions after scaling
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        # 计算需要padding的像素数 / Calculate pixels needed for padding
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        # 四舍五入计算padding值 / Round padding values
        top = (int)(round(dh - 0.1))
        bottom = (int)(round(dh + 0.1))
        left = (int)(round(dw - 0.1))
        right = (int)(round(dw + 0.1))
        return top, bottom, left, right

    def get_color(self, x):
        """
        根据类别索引获取颜色
        Get color based on class index

        参数：
        Parameters:
            x: 类别索引 / Class index

        返回：
        Returns:
            color: 颜色值 / Color value
        """
        idx = x % len(self.color_four)  # 循环使用颜色 / Cycle through colors
        return self.color_four[idx]


if __name__ == "__main__":
    ALERT_LABELS = ("person", "car")
    BUZZER_FREQUENCY = 2000
    BUZZER_VOLUME = 50
    BUZZER_DURATION = 0.1
    BUZZER_INTERVAL_MS = 300

    # 显示模式，默认"hdmi"，可以选择"hdmi"和"lcd"，k230d受限于内存不支持
    # Display mode, default "hdmi", can choose between "hdmi" and "lcd", k230d does not support due to memory limitations
    display_mode = "lcd"
    if display_mode == "hdmi":
        display_size = [1920, 1080]  # HDMI显示分辨率 / HDMI display resolution
    else:
        display_size = [640, 480]    # LCD显示分辨率 / LCD display resolution

    # 模型路径 / Model path
    kmodel_path = "/sdcard/kmodel/yolov8n_seg_320.kmodel"
    # 80个COCO数据集类别标签 / 80 COCO dataset class labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
             "scissors", "teddy bear", "hair drier", "toothbrush"]

    # 其它参数设置 / Other parameter settings
    confidence_threshold = 0.2  # 置信度阈值 / Confidence threshold
    nms_threshold = 0.5        # NMS阈值 / NMS threshold
    mask_threshold = 0.5       # 掩码阈值 / Mask threshold
    rgb888p_size = [320, 320]  # 预处理后的图像尺寸 / Image size after preprocessing

    # 初始化PipeLine / Initialize PipeLine
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()
    buzzer = BuzzerAlert(
        frequency=BUZZER_FREQUENCY,
        volume=BUZZER_VOLUME,
        duration=BUZZER_DURATION,
        interval_ms=BUZZER_INTERVAL_MS
    )

    # 初始化自定义YOLOV8分割示例 / Initialize custom YOLOV8 segmentation example
    seg = SegmentationApp(
        kmodel_path,
        labels=labels,
        model_input_size=[320, 320],
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        mask_threshold=mask_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0,
        alert_labels=ALERT_LABELS
    )

    # 配置预处理 / Configure preprocessing
    seg.config_preprocess()

    # 主循环 / Main loop
    try:
        while True:
            with ScopedTiming("total", 1):
                # 获取当前帧数据 / Get current frame data
                img = pl.get_frame()
                # 推理当前帧 / Inference on current frame
                seg_res = seg.run(img)
                buzzer.update(seg.has_alert_target(seg_res))
                # 绘制结果到PipeLine的osd图像 / Draw results to PipeLine's osd image
                seg.draw_result(pl, seg_res)
                # 显示当前的绘制结果 / Display current drawing results
                pl.show_image()
                # 垃圾回收，避免内存泄漏 / Garbage collection to avoid memory leaks
                gc.collect()
    finally:
        # 释放资源 / Release resources
        buzzer.off()
        seg.deinit()
        pl.destroy()
