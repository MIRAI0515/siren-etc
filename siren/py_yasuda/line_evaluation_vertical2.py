from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.ndimage import zoom  # スケーリングのために追加

# 画像のパスを指定
input_image_path1 = '/mnt/siren/py_yasuda/gradient_image_pattern_16x16_128_252_372.tiff' # こっちを基準とする
input_image_path2 = '/mnt/siren/explore_siren/trained_siren_240613_2204_EGFP_m4096_s128_252_372/512_model_output_processed.tiff'  # 画像のパスに変更

# 画像を開く
image1 = Image.open(input_image_path1)
image2 = Image.open(input_image_path2)
# 画像をグレースケールに変換
image_gray1 = image1.convert("L")
image_gray2 = image2.convert("L")
image_np2 = np.array(image_gray2) # numpy配列に変換

# 2つ目の画像の輝度値を修正する関数
def process_numpy(data):
    rows, cols = data.shape
    processed_data = data.copy()

    # データの最小値を取得
    min_value = np.min(data)
    correction_value = min_value - 0.001  # 修正値を最小値から-0.001とする

    # マトリックス内の全要素に対して、周囲と比較（境界は無視）
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (data[i, j] < data[i - 1, j] and  # 上
                data[i, j] < data[i + 1, j] and  # 下
                data[i, j] < data[i, j - 1] and  # 左
                data[i, j] < data[i, j + 1]):    # 右
                processed_data[i, j] = correction_value  # 修正された値を設定

    return processed_data

# 画像2の輝度値を修正
processed_image2 = process_numpy(image_np2)
image_np1 = np.array(image_gray1)

# 画像サイズの比率を計算
scaling_factor = processed_image2.shape[0] / image_np1.shape[0]  # image1に対してimage2のサイズ比率を計算
print(f"image_np1.shape[0]:{image_np1.shape[0]}")
print(f"processed_image2.shape[0]:{processed_image2.shape[0]}")
print(f"scaling_factor:{scaling_factor}")

# 画像表示用のセットアップ（横に並べて表示）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 横に2つの画像を並べる
ax1.imshow(image_np1, cmap='gray')
ax1.set_title('Image 1')
ax2.imshow(processed_image2, cmap='gray')
ax2.set_title('Image 2 (Processed with Highlighted Modifications)')

line_plot1, = ax1.plot([], [], 'r', linewidth=1.5)  # 画像1用の赤線プロット
line_plot2, = ax2.plot([], [], 'r', linewidth=1.5)  # 画像2用の赤線プロット

# 輝度グラフのセットアップ
fig_brightness, ax_brightness = plt.subplots()
line_brightness1, = ax_brightness.plot([], [], label="original")  # 輝度値グラフのライン
line_brightness2, = ax_brightness.plot([], [], label="Image super-resolved by SIREN")  # 2つ目の画像の輝度値グラフ
ax_brightness.set_xlabel("Column (Pixel)")
ax_brightness.set_ylabel("Brightness (0-255)")
ax_brightness.grid(True)
ax_brightness.legend()

# 赤い縦線を更新する関数
def update_red_lines(col):
    # 赤線を追加または更新
    line_plot1.set_data([col, col], [0, image_np1.shape[0]])  # Image 1 に赤線
    line_plot2.set_data([col * scaling_factor, col * scaling_factor], [0, processed_image2.shape[0]])  # Image 2 に赤線
    fig.canvas.draw()  # 描画を更新

# 輝度を縦方向に更新する関数（画像1に対して）
def update_brightness_vertical(row, col, image_np, scaling_factor=1):
    row = int(row * scaling_factor)  # スケーリングに合わせて座標を変換
    col = int(col * scaling_factor)

    if 0 <= col < image_np.shape[1]:  # 列が範囲内にあるかチェック
        brightness_values = image_np[:, col]  # 縦方向（行ごと）の輝度値を取得
        return brightness_values

# 縦方向の輝度グラフを更新する関数
def update_brightness_graph_vertical(row, col):
    # 画像1の縦方向の輝度値を取得
    brightness_values1 = update_brightness_vertical(row, col, image_np1)
    
    # 画像2の縦方向の輝度値を取得（座標とピクセル数を調整）
    brightness_values2 = update_brightness_vertical(row, col, processed_image2, scaling_factor)
    
    # ピクセル数が小さい方のデータをスケーリングして、輝度値を同じ長さに合わせる
    if len(brightness_values1) < len(brightness_values2):
        brightness_values1 = zoom(brightness_values1, scaling_factor)  # 小さい方の輝度値を拡張
    else:
        brightness_values2 = zoom(brightness_values2, 1/scaling_factor)  # 逆に拡張

    # グラフを更新
    line_brightness1.set_data(np.arange(len(brightness_values1)), brightness_values1)
    line_brightness2.set_data(np.arange(len(brightness_values2)), brightness_values2)
    
    ax_brightness.set_xlim(0, max(len(brightness_values1), len(brightness_values2)))
    ax_brightness.set_ylim(0, 255)
    ax_brightness.set_title(f"Col {col} Brightness Comparison (Vertical)")
    fig_brightness.canvas.draw()

    # 赤線を更新して、どこをクリックしたか示す
    update_red_lines(col)

# クリックイベントで縦軸のグラフを表示する場合
def on_click_vertical(event):
    if event.inaxes in [ax1, ax2]:  # どちらの画像でもクリック可能
        row = int(event.ydata)
        col = int(event.xdata)
        update_brightness_graph_vertical(row, col)
        print(f"クリックされた座標: 行={row}, 列={col}")

# テキストボックスから座標を取得し、Pushボタンでグラフを更新
def submit_coordinates_vertical(event):
    try:
        row = int(text_box_row.text)
        col = int(text_box_col.text)
        update_brightness_graph_vertical(row, col)
    except ValueError:
        print("有効な行番号と列番号を入力してください")

# テキストボックスのセットアップ
fig_textbox, (ax_textbox_row, ax_textbox_col) = plt.subplots(1, 2, figsize=(8, 1))

text_box_row = TextBox(ax_textbox_row, "Row:", initial="0")
text_box_col = TextBox(ax_textbox_col, "Col:", initial="0")

# Pushボタンを追加
ax_button_vertical = plt.axes([0.8, 0.05, 0.1, 0.075])  # ボタンの位置
button_vertical = Button(ax_button_vertical, 'Push Vertical')
button_vertical.on_clicked(submit_coordinates_vertical)  # ボタンクリック時にsubmit_coordinatesを呼び出す

# マウスクリックイベントでの縦線追加
fig.canvas.mpl_connect('button_press_event', on_click_vertical)

plt.show()
