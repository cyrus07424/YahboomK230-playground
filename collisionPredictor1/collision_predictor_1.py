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
from ybUtils.YbKey import YbKey
from ybUtils.YbRGB import YbRGB

# カスタムYOLOv8セグメンテーションクラス Custom YOLOv8 Segmentation Class


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


class AlertToggleButton:
    def __init__(self, initial_enabled=False, debounce_ms=250):
        self.key = YbKey()
        self.enabled = initial_enabled
        self.debounce_ms = debounce_ms
        self.was_pressed = False
        self.last_toggle_ms = utime.ticks_add(utime.ticks_ms(), -self.debounce_ms)

    def update(self):
        pressed = self.key.is_pressed()
        now_ms = utime.ticks_ms()
        if pressed and not self.was_pressed and utime.ticks_diff(now_ms, self.last_toggle_ms) >= self.debounce_ms:
            self.enabled = not self.enabled
            self.last_toggle_ms = now_ms
        self.was_pressed = pressed
        return self.enabled


class LedIndicator:
    OFF = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    def __init__(self):
        self.led = YbRGB()
        self.current_color = None
        self.off()

    def update(self, vehicle_detected, collision_risk):
        if collision_risk:
            self.show(self.RED)
        elif vehicle_detected:
            self.show(self.GREEN)
        else:
            self.show(self.OFF)

    def off(self):
        self.show(self.OFF)

    def show(self, color):
        if self.current_color != color:
            self.led.show_rgb(color)
            self.current_color = color


class SegmentationApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size, confidence_threshold=0.2, nms_threshold=0.5, mask_threshold=0.5, rgb888p_size=[224, 224], display_size=[1920, 1080], debug_mode=0, alert_labels=("person", "bicycle", "car", "motorcycle", "bus", "truck"), vehicle_labels=("bicycle", "car", "motorcycle", "bus", "truck", "train", "boat")):
        """
        セグメンテーションアプリケーションクラスを初期化する
        Initialize the segmentation application class

        引数:
        Parameters:
            kmodel_path: モデルファイルのパス / Path to the kmodel file
            labels: クラスラベルのリスト / List of class labels
            model_input_size: モデル入力サイズ / Model input size
            confidence_threshold: 信頼度しきい値 / Confidence threshold
            nms_threshold: 非極大値抑制のしきい値 / Non-maximum suppression threshold
            mask_threshold: マスクしきい値 / Mask threshold
            rgb888p_size: RGB画像サイズ / RGB image size
            display_size: 表示サイズ / Display size
            debug_mode: デバッグモード / Debug mode
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        # モデルパス / Model path
        self.kmodel_path = kmodel_path
        # セグメンテーションクラスラベル / Segmentation class labels
        self.labels = labels
        self.alert_label_ids = set()
        for label in alert_labels:
            if label in self.labels:
                self.alert_label_ids.add(self.labels.index(label))
        self.vehicle_label_ids = set()
        for label in vehicle_labels:
            if label in self.labels:
                self.vehicle_label_ids.add(self.labels.index(label))
        # モデル入力解像度 / Model input resolution
        self.model_input_size = model_input_size
        # 信頼度しきい値 / Confidence threshold
        self.confidence_threshold = confidence_threshold
        # NMSしきい値 / NMS threshold
        self.nms_threshold = nms_threshold
        # マスクしきい値 / Mask threshold
        self.mask_threshold = mask_threshold
        # センサーからAIに渡す画像解像度（幅は16の倍数に揃える）
        # Image resolution from sensor to AI (width aligned to multiple of 16)
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 表示解像度（幅は16の倍数に揃える）
        # Display resolution (width aligned to multiple of 16)
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        # 検出ボックス用のプリセット色 (ARGB形式: Alpha, Red, Green, Blue)
        # Preset colors for detection boxes (ARGB format: Alpha, Red, Green, Blue)
        self.color_four = [(255, 220, 20, 60), (255, 119, 11, 32), (255, 0, 0, 142), (255, 0, 0, 230),
                         (255, 106, 0, 228), (255, 0, 60, 100), (255, 0, 80, 100), (255, 0, 0, 70),
                         (255, 0, 0, 192), (255, 250, 170, 30), (255, 100, 170, 30), (255, 220, 220, 0),
                         (255, 175, 116, 175), (255, 250, 0, 30), (255, 165, 42, 42), (255, 255, 77, 255),
                         (255, 0, 226, 252), (255, 182, 182, 255), (255, 0, 82, 0), (255, 120, 166, 157)]
        # aidemoの後処理インターフェースに渡すセグメンテーション結果のnumpy.array
        # Numpy array for segmentation results, used for the aidemo post-processing interface
        self.masks = np.zeros((1, self.display_size[1], self.display_size[0], 4))
        # モデル前処理を行うAi2dインスタンス
        # Ai2d instance for model preprocessing
        self.ai2d = Ai2d(debug_mode)
        # Ai2dの入出力形式と型を設定
        # Set Ai2d input and output formats and types
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """
        前処理を設定する
        Configure preprocessing operations

        引数:
        Parameters:
            input_image_size: 入力画像サイズ。Noneの場合はデフォルトサイズを使用
                            Input image size, if None, use default size
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # ai2d前処理設定を初期化。デフォルトはセンサーからAIへ渡すサイズで、
            # input_image_sizeを設定すると入力サイズを変更できる
            # Initialize ai2d preprocessing configuration, default is the size from sensor to AI,
            # you can modify the input size by setting input_image_size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            # アスペクト比を保つようにpadding処理を設定
            # Configure padding operation to maintain aspect ratio
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            # 双線形補間を使うresize処理を設定
            # Configure resize operation using bilinear interpolation
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # ai2d処理グラフを構築
            # Build ai2d processing graph
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        """
        このタスク向けの後処理を行う
        Custom post-processing for the current task

        引数:
        Parameters:
            results: モデル推論結果 / Model inference results

        戻り値:
        Returns:
            seg_res: セグメンテーション結果 / Segmentation results
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # ここではaidemoのsegment_postprocessインターフェースを使って後処理を行う
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

    def draw_result(self, pl, seg_res, buzzer_enabled=False):
        """
        セグメンテーション結果を表示レイヤーに描画する
        Draw segmentation results to display layer

        引数:
        Parameters:
            pl: Pipelineオブジェクト / Pipeline object
            seg_res: セグメンテーション結果 / Segmentation results
        """
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if seg_res[0]:  # 物体が検出された場合 / If objects are detected
                pl.osd_img.clear()  # OSDレイヤーをクリア / Clear OSD layer
                # maskデータを参照する画像オブジェクトを作成 / Create image object referencing mask data
                mask_img = image.Image(self.display_size[0], self.display_size[1], image.ARGB8888, alloc=image.ALLOC_REF, data=self.masks)
                pl.osd_img.copy_from(mask_img)  # mask画像をOSDレイヤーへコピー / Copy mask image to OSD layer

                # 検出結果を取り出す / Extract detection results
                dets, ids, scores = seg_res[0], seg_res[1], seg_res[2]
                for i, det in enumerate(dets):
                    # ラベルと信頼度を描画 / Draw label and confidence
                    x1, y1, w, h = map(lambda x: int(round(x, 0)), det)
                    pl.osd_img.draw_string_advanced(
                        x1, y1-50, 32,
                        " " + self.labels[int(ids[i])] + " " + str(round(scores[i], 2)),
                        color=self.get_color(int(ids[i]))
                    )
            else:
                pl.osd_img.clear()  # 検出結果がない場合はOSDをクリア / Clear OSD when no detection

            self.draw_buzzer_indicator(pl, buzzer_enabled)

    def draw_buzzer_indicator(self, pl, buzzer_enabled):
        indicator_text = "BZ ON" if buzzer_enabled else "BZ OFF"
        indicator_color = (255, 80, 220, 80) if buzzer_enabled else (255, 220, 80, 80)
        indicator_x = self.display_size[0] - 100
        indicator_y = 10
        pl.osd_img.draw_string_advanced(indicator_x, indicator_y, 24, indicator_text, color=indicator_color)

    def is_collision_risk(self, det, class_id):
        x1, y1, w, h = det
        frame_w = float(self.display_size[0])
        frame_h = float(self.display_size[1])
        center_x = x1 + (w / 2.0)
        bottom_y = y1 + h
        width_ratio = w / frame_w
        height_ratio = h / frame_h
        area_ratio = (w * h) / (frame_w * frame_h)
        center_offset_ratio = abs(center_x - (frame_w / 2.0)) / frame_w
        bottom_ratio = bottom_y / frame_h
        label = self.labels[class_id]

        if label == "person":
            large_enough = height_ratio >= 0.22 or area_ratio >= 0.035
        else:
            large_enough = width_ratio >= 0.16 or height_ratio >= 0.18 or area_ratio >= 0.05

        in_driving_path = center_offset_ratio <= 0.22 or (center_offset_ratio <= 0.30 and width_ratio >= 0.18)
        close_enough = bottom_ratio >= 0.60
        return in_driving_path and close_enough and large_enough

    def has_alert_target(self, seg_res):
        if not seg_res[0]:
            return False

        dets, ids = seg_res[0], seg_res[1]
        for det, class_id in zip(dets, ids):
            class_id = int(class_id)
            if class_id in self.alert_label_ids and self.is_collision_risk(det, class_id):
                return True
        return False

    def has_vehicle_target(self, seg_res):
        if not seg_res[0]:
            return False

        for class_id in seg_res[1]:
            if int(class_id) in self.vehicle_label_ids:
                return True
        return False

    def get_padding_param(self):
        """
        アスペクト比を保つためのpaddingパラメータを計算する
        Calculate padding parameters to maintain aspect ratio

        戻り値:
        Returns:
            top, bottom, left, right: paddingパラメータ / padding parameters
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        # 拡大縮小率を計算 / Calculate scaling ratio
        ratio_w = float(dst_w) / self.rgb888p_size[0]
        ratio_h = float(dst_h) / self.rgb888p_size[1]
        # 画像全体が収まるよう、より小さい比率を選ぶ / Choose smaller ratio to ensure entire image fits
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        # 拡大縮小後の新しいサイズを計算 / Calculate new dimensions after scaling
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        # 必要なpaddingピクセル数を計算 / Calculate pixels needed for padding
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        # 四捨五入してpadding値を計算 / Round padding values
        top = (int)(round(dh - 0.1))
        bottom = (int)(round(dh + 0.1))
        left = (int)(round(dw - 0.1))
        right = (int)(round(dw + 0.1))
        return top, bottom, left, right

    def get_color(self, x):
        """
        クラスインデックスに応じた色を取得する
        Get color based on class index

        引数:
        Parameters:
            x: クラスインデックス / Class index

        戻り値:
        Returns:
            color: 色の値 / Color value
        """
        idx = x % len(self.color_four)  # 色を循環利用 / Cycle through colors
        return self.color_four[idx]


if __name__ == "__main__":
    ALERT_LABELS = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
    BUZZER_FREQUENCY = 2000
    BUZZER_VOLUME = 50
    BUZZER_DURATION = 0.1
    BUZZER_INTERVAL_MS = 300
    ALERTS_ENABLED_AT_START = False

    # 表示モード。デフォルトは"hdmi"で、"hdmi"と"lcd"を選択可能。k230dはメモリ制約のため非対応
    # Display mode, default "hdmi", can choose between "hdmi" and "lcd", k230d does not support due to memory limitations
    display_mode = "lcd"
    if display_mode == "hdmi":
        display_size = [1920, 1080]  # HDMI表示解像度 / HDMI display resolution
    else:
        display_size = [640, 480]    # LCD表示解像度 / LCD display resolution

    # モデルパス / Model path
    kmodel_path = "/sdcard/kmodel/yolov8n_seg_320.kmodel"
    # COCOデータセットの80クラスラベル / 80 COCO dataset class labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
             "scissors", "teddy bear", "hair drier", "toothbrush"]

    # その他のパラメータ設定 / Other parameter settings
    confidence_threshold = 0.2  # 信頼度しきい値 / Confidence threshold
    nms_threshold = 0.5        # NMSしきい値 / NMS threshold
    mask_threshold = 0.5       # マスクしきい値 / Mask threshold
    rgb888p_size = [320, 320]  # 前処理後の画像サイズ / Image size after preprocessing

    # PipeLineを初期化 / Initialize PipeLine
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()
    buzzer = BuzzerAlert(
        frequency=BUZZER_FREQUENCY,
        volume=BUZZER_VOLUME,
        duration=BUZZER_DURATION,
        interval_ms=BUZZER_INTERVAL_MS
    )
    led = LedIndicator()
    alert_toggle = AlertToggleButton(initial_enabled=ALERTS_ENABLED_AT_START)

    # カスタムYOLOV8セグメンテーションのサンプルを初期化 / Initialize custom YOLOV8 segmentation example
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

    # 前処理を設定 / Configure preprocessing
    seg.config_preprocess()

    # メインループ / Main loop
    try:
        while True:
            with ScopedTiming("total", 1):
                alerts_enabled = alert_toggle.update()
                # 現在のフレームデータを取得 / Get current frame data
                img = pl.get_frame()
                # 現在のフレームで推論 / Inference on current frame
                seg_res = seg.run(img)
                collision_risk = seg.has_alert_target(seg_res)
                buzzer.update(alerts_enabled and collision_risk)
                led.update(seg.has_vehicle_target(seg_res), collision_risk)
                # 結果をPipeLineのosd画像に描画 / Draw results to PipeLine's osd image
                seg.draw_result(pl, seg_res, buzzer_enabled=alerts_enabled)
                # 現在の描画結果を表示 / Display current drawing results
                pl.show_image()
                # ガベージコレクションでメモリリークを防止 / Garbage collection to avoid memory leaks
                gc.collect()
    finally:
        # リソースを解放 / Release resources
        buzzer.off()
        led.off()
        seg.deinit()
        pl.destroy()
