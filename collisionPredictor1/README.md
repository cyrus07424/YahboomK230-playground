# Collision Predictor ver.1

Yahboom K230 向けの衝突危険度可視化スクリプトです。  
カメラ映像を YOLOv8 セグメンテーションモデルで解析し、検出結果に応じて画面表示・ブザー・RGB LED を制御します。

## 対応機種

- Yahboom K230

## 概要

このスクリプトは `person` / `bicycle` / `car` / `motorcycle` / `bus` / `truck` などを検出し、物体の大きさ・画面中央への近さ・画面下端への近さから危険度を算出します。

危険度は 0.0〜1.0 の連続値として扱われ、RGB LED は次のように変化します。

- 車両・乗り物が見えていて危険度が低い: 緑
- 危険度が上がる: 緑から黄を経由して赤へ変化
- 車両・乗り物が見えていない: 消灯
- 衝突危険が高い対象がある: ブザー設定に関係なく赤寄りに点灯

ブザーは LED と独立しており、ボタンで ON/OFF を切り替えられます。

## 主な機能

- YOLOv8 セグメンテーションによる対象検出
- 検出結果の OSD 表示
- 危険度スコアの連続判定
- RGB LED のグラデーション表示
- 危険時のブザー通知
- ボタンによるブザー有効/無効切り替え

## 必要ファイル

- `collision_predictor_1.py`
- モデルファイル: `/sdcard/kmodel/yolov8n_seg_320.kmodel`

## 使用ライブラリ

このスクリプトは Yahboom K230 環境に含まれる以下のライブラリを利用します。

- `libs.PipeLine`
- `libs.AIBase`
- `libs.AI2D`
- `aidemo`
- `ybUtils.YbBuzzer`
- `ybUtils.YbKey`
- `ybUtils.YbRGB`

## 実行前の準備

1. `collision_predictor_1.py` を K230 に配置します。
2. `yolov8n_seg_320.kmodel` を `/sdcard/kmodel/` に配置します。
3. 本体のカメラ、RGB LED、ブザー、キーが使える状態で起動します。

## 実行方法

CanMV IDE などで `collision_predictor_1.py` を開いて実行してください。

## 動作仕様

### 表示

- セグメンテーション結果を OSD に重ねて表示します。
- 右上に `BZ ON` / `BZ OFF` が表示されます。

### ブザー

- 起動時は `OFF` です。
- 本体キーで ON/OFF を切り替えます。
- ON のときのみ、高危険度判定で断続的に鳴動します。

### RGB LED

- `YbRGB.show_rgb()` を使って本体 LED を制御します。
- 車両・乗り物が見えている間は危険度に応じて緑→黄→赤に変化します。
- 人などが高危険度で接近した場合も、ブザーが OFF でも LED は赤寄りに変化します。
- 対象がいなければ消灯します。

## 主な設定項目

スクリプト末尾の定数を変更することで挙動を調整できます。

| 項目 | 内容 |
| --- | --- |
| `display_mode` | `lcd` または `hdmi` |
| `kmodel_path` | 使用する kmodel のパス |
| `confidence_threshold` | 検出信頼度しきい値 |
| `nms_threshold` | NMS しきい値 |
| `mask_threshold` | マスクしきい値 |
| `BUZZER_FREQUENCY` | ブザー周波数 |
| `BUZZER_VOLUME` | ブザー音量 |
| `BUZZER_DURATION` | 1 回の鳴動時間 |
| `BUZZER_INTERVAL_MS` | ブザーの繰り返し間隔 |
| `ALERTS_ENABLED_AT_START` | 起動時のブザー有効状態 |

## 危険度判定について

危険度は単純なしきい値ではなく、以下の要素から連続的に計算されます。

- 検出物体のサイズ
- 画面中央付近にいるか
- 画面下側まで接近しているか

このスコアをもとに LED 色を補間し、一定以上のスコアを超えた場合のみブザーを鳴らします。

## 注意

- `display_mode = "lcd"` が現在のデフォルトです。
- K230 上のライブラリ構成やモデル配置先が異なる場合は、パスを環境に合わせて変更してください。
- 危険度の重みやしきい値はカメラ取り付け位置や用途に応じて調整してください。
