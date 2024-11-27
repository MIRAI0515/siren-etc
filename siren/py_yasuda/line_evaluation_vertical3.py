from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.ndimage import zoom  # スケーリングのために追加

# 画像のパスを指定
input_image_path1 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff'  # 1つ目の画像
input_image_path2 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_512x512_no_resize.tiff'  # 2つ目の画像
input_image_path3 = '/mnt/siren/explore_siren/trained_siren_240613_2204_EGFP_resized_512x512_no_resize_1024/1024_model_output.tiff'  # 3つ目の画像
pixel = 128  # 変更

# 画像を開く
image1 = Image.open(input_image_path1)
image2 = Image.open(input_image_path2)
image3 = Image.open(input_image_path3)
# 画像をグレースケールに変換
image_gray1 = image1.convert("L")
image_gray2 = image2.convert("L")
image_gray3 = image3.convert("L")
image_np3 = np.array(image_gray3) # numpy配列に変換

"""
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
"""

# 画像2の輝度値を修正
image_np1 = np.array(image_gray1)
image_np2 = np.array(image_gray2)
image_np3 = np.array(image_gray3)
# processed_image3 = process_numpy(image_np3)

# 画像サイズの比率を計算
scaling_factor1 = image_np2.shape[0] / image_np1.shape[0]  # image1に対してimage2のサイズ比率を計算
# scaling_factor2 = processed_image3.shape[0] / image_np1.shape[0]  # image1に対してimage2のサイズ比率を計算
scaling_factor2 = image_np3.shape[0] / image_np1.shape[0] 
print(f"image_np1.shape[0]:{image_np1.shape[0]}")
print(f"image_np2.shape[0]:{image_np2.shape[0]}")
print(f"image_np3.shape[0]:{image_np3.shape[0]}")
# print(f"processed_image3.shape[0]:{processed_image3.shape[0]}")
print(f"scaling_factor1:{scaling_factor1}")
print(f"scaling_factor2:{scaling_factor2}")

# 画像表示用のセットアップ（横に並べて表示）
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # 横に2つの画像を並べる
ax1.imshow(image_np1, cmap='gray')
ax1.set_title('Image 1')
ax2.imshow(image_np2, cmap='gray')
ax2.set_title('Image 2')
# ax3.imshow(processed_image3, cmap='gray')
ax3.imshow(image_np3, cmap='gray')
ax3.set_title('Image 3 (Processed with Highlighted Modifications)')

line_plot1, = ax1.plot([], [], 'r', linewidth=1.5)  # 画像1用の赤線プロット
line_plot2, = ax2.plot([], [], 'r', linewidth=1.5)  # 画像2用の赤線プロット
line_plot3, = ax3.plot([], [], 'r', linewidth=1.5)  # 画像3用の赤線プロット

# 輝度グラフのセットアップ
fig_brightness, ax_brightness = plt.subplots()
line_brightness1, = ax_brightness.plot([], [], label="original")  # 輝度値グラフのライン
line_brightness2, = ax_brightness.plot([], [], label="resized_512x512")  # 輝度値グラフのライン
line_brightness3, = ax_brightness.plot([], [], label="Image super-resolved by SIREN")  # 3つ目の画像の輝度値グラフ
ax_brightness.set_xlabel("Column (Pixel)")
ax_brightness.set_ylabel("Brightness (0-255)")
ax_brightness.grid(True)
ax_brightness.legend()

# 赤い縦線を更新する関数
def update_red_lines(col):
    # 赤線を追加または更新
    line_plot1.set_data([col, col], [0, image_np1.shape[0]])  # Image 1 に赤線
    line_plot2.set_data([col * scaling_factor1, col * scaling_factor1], [0, image_np2.shape[0]])  # Image 2 に赤線
    # line_plot3.set_data([col * scaling_factor2, col * scaling_factor2], [0, processed_image3.shape[0]])  # Image 2 に赤線
    line_plot3.set_data([col * scaling_factor2, col * scaling_factor2], [0, image_np3.shape[0]])
    fig.canvas.draw()  # 描画を更新

# 輝度を縦方向に更新する関数（画像1に対して）
def update_brightness_vertical(row, col, image_np, pixel, scaling_factor=1):
    row = int(row * scaling_factor)  # スケーリングに合わせて座標を変換
    col = int(col * scaling_factor)
    pixel = int(pixel * scaling_factor)

    if 0 <= row < image_np.shape[0]:  # 行が範囲内にあるかチェック
        start_col = max(0, col - pixel // 2)
        end_col = min(image_np.shape[1], start_col + pixel)
        start_row = max(0, row - pixel // 2)
        end_row = min(image_np.shape[0], start_row + pixel)

        brightness_values = image_np[start_row:end_row, col]  # 指定された行の輝度値を取得

        # 抽出した部分を返す
        cropped_image = image_np[start_row:end_row, start_col:end_col]
        
        return brightness_values, cropped_image

# 縦方向の輝度グラフを更新する関数
def update_brightness_graph_vertical(row, col):
    # 画像1の縦方向の輝度値を取得
    brightness_values1, cropped_image1 = update_brightness_vertical(row, col, image_np1, pixel)
    
    # 画像2の縦方向の輝度値を取得（座標とピクセル数を調整）
    brightness_values2, cropped_image2 = update_brightness_vertical(row, col, image_np2, pixel, scaling_factor1)
    
    # 画像2の縦方向の輝度値を取得（座標とピクセル数を調整）
    # brightness_values3, cropped_image3 = update_brightness_vertical(row, col, processed_image3, pixel, scaling_factor2)
    brightness_values3, cropped_image3 = update_brightness_vertical(row, col, image_np3, pixel, scaling_factor2)
    
    # ピクセル数が小さい方のデータをスケーリングして、輝度値を同じ長さに合わせる
    max_length = max(len(brightness_values1), len(brightness_values2), len(brightness_values3))
    if len(brightness_values1) < max_length:
        brightness_values1 = zoom(brightness_values1, max_length / len(brightness_values1))  # 1つ目の輝度値を拡張
    if len(brightness_values2) < max_length:
        brightness_values2 = zoom(brightness_values2, max_length / len(brightness_values2))  # 2つ目の輝度値を拡張
    if len(brightness_values3) < max_length:
        brightness_values3 = zoom(brightness_values3, max_length / len(brightness_values3))  # 3つ目の輝度値を拡張


    # グラフを更新
    line_brightness1.set_data(np.arange(len(brightness_values1)), brightness_values1)
    line_brightness2.set_data(np.arange(len(brightness_values2)), brightness_values2)
    line_brightness3.set_data(np.arange(len(brightness_values3)), brightness_values3)
    
    ax_brightness.set_xlim(0, max(len(brightness_values1), len(brightness_values2), len(brightness_values3)))
    ax_brightness.set_ylim(0, 255)
    ax_brightness.set_title(f"Row {row} Col {col} Brightness Comparison")
    fig_brightness.canvas.draw()
    
    # 抽出された部分を並べて表示
    fig_crop, (ax_crop1, ax_crop2, ax_crop3) = plt.subplots(1, 3, figsize=(12, 4))  # 横に並べて表示
    ax_crop1.imshow(cropped_image1, cmap='gray')
    ax_crop1.set_title(f"Image 1 - Row {row}, Col {col}")
    # ax_crop1.axvline(x=pixel//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop1.axvline(x=97, color='red', linestyle='-', linewidth=1.5)
    
    ax_crop2.imshow(cropped_image2, cmap='gray')
    ax_crop2.set_title(f"Image 2 - Row {int(row*scaling_factor1)}, Col {int(col*scaling_factor1)}")
    # ax_crop2.axvline(x=pixel*scaling_factor1//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop2.axvline(x=97*scaling_factor1, color='red', linestyle='-', linewidth=1.5)  
    
    ax_crop3.imshow(cropped_image3, cmap='gray')
    ax_crop3.set_title(f"Image 3 - Row {int(row*scaling_factor2)}, Col {int(col*scaling_factor2)}")
    # ax_crop3.axvline(x=pixel*scaling_factor2//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop3.axvline(x=97*scaling_factor2, color='red', linestyle='-', linewidth=1.5) 
    
    plt.show()

    # 赤線を更新して、どこをクリックしたか示す
    # update_red_lines(col)

# クリックイベントで縦軸のグラフを表示する場合
def on_click_vertical(event):
    if event.inaxes in [ax1, ax2, ax3]:  # どちらの画像でもクリック可能
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
