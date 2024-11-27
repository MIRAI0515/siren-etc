from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.ndimage import zoom  # スケーリングのために追加

# 画像のパスを指定
input_image_path1 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff' # こっちを基準とする
input_image_path2 = '/mnt/siren/explore_siren/trained_siren_240613_2204_EGFP_2048/2048_model_output.tiff'  # 画像のパスに変更

pixel = 128  # 変更

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
            # 上下左右の要素よりも小さい場合、その値を条件に応じて変更
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
line_brightness1, = ax_brightness.plot([], [], label="Image 1 Brightness")  # 輝度値グラフのライン
line_brightness2, = ax_brightness.plot([], [], label="Image 2 Brightness")  # 2つ目の画像の輝度値グラフ
ax_brightness.set_xlabel("Column (Pixel)")
ax_brightness.set_ylabel("Brightness (0-255)")
ax_brightness.grid(True)
ax_brightness.legend()

# 行の輝度を更新する関数（画像1に対して）
def update_brightness(row, col, image_np, pixel, scaling_factor=1):
    row = int(row * scaling_factor)  # スケーリングに合わせて座標を変換
    col = int(col * scaling_factor)
    pixel = int(pixel * scaling_factor)

    if 0 <= row < image_np.shape[0]:  # 行が範囲内にあるかチェック
        start_col = max(0, col - pixel // 2)
        end_col = min(image_np.shape[1], start_col + pixel)
        start_row = max(0, row - pixel // 2)
        end_row = min(image_np.shape[0], start_row + pixel)

        brightness_values = image_np[row, start_col:end_col]  # 指定された行の輝度値を取得

        # 抽出した部分を返す
        cropped_image = image_np[start_row:end_row, start_col:end_col]
        
        return brightness_values, cropped_image

# 輝度グラフを更新し、抽出画像を並べて表示する関数
def update_brightness_graph(row, col):
    # 画像1の輝度値と抽出された部分を取得
    brightness_values1, cropped_image1 = update_brightness(row, col, image_np1, pixel)
    
    # 画像2の輝度値と抽出された部分を取得（座標とピクセル数を調整）
    brightness_values2, cropped_image2 = update_brightness(row, col, processed_image2, pixel, scaling_factor)
    
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
    ax_brightness.set_title(f"Row {row} Col {col} Brightness Comparison")
    fig_brightness.canvas.draw()

    # 抽出された部分を並べて表示
    fig_crop, (ax_crop1, ax_crop2) = plt.subplots(1, 2, figsize=(8, 4))  # 横に並べて表示
    ax_crop1.imshow(cropped_image1, cmap='gray')
    ax_crop1.set_title(f"Image 1 - Row {row}, Col {col}")
    ax_crop1.axhline(y=pixel//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    
    ax_crop2.imshow(cropped_image2, cmap='gray')
    ax_crop2.set_title(f"Image 2 - Row {int(row*scaling_factor)}, Col {int(col*scaling_factor)}")
    ax_crop2.axhline(y=pixel*scaling_factor//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    
    plt.show()


# クリックイベントハンドラ
def on_click(event):
    if event.inaxes in [ax1, ax2]:  # どちらの画像でもクリック可能
        row = int(event.ydata)
        col = int(event.xdata)
        update_brightness_graph(row, col)
        print(f"クリックされた座標: 行={row}, 列={col}")


# テキストボックスから座標を取得し、Pushボタンでグラフを更新
def submit_coordinates(event):
    try:
        row = int(text_box_row.text)
        col = int(text_box_col.text)
        update_brightness_graph(row, col)
    except ValueError:
        print("有効な行番号と列番号を入力してください")

# テキストボックスのセットアップ
fig_textbox, (ax_textbox_row, ax_textbox_col) = plt.subplots(1, 2, figsize=(8, 1))

text_box_row = TextBox(ax_textbox_row, "Row:", initial="0")
text_box_col = TextBox(ax_textbox_col, "Col:", initial="0")

# Pushボタンを追加
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # ボタンの位置
button = Button(ax_button, 'Push')
button.on_clicked(submit_coordinates)  # ボタンクリック時にsubmit_coordinatesを呼び出す

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()








