from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.ndimage import zoom  # スケーリングのために追加

# 画像のパスを指定
# input_image_path1 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff'  # 1つ目の画像
# input_image_path2 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_512x512_no_resize.tiff'  # 2つ目の画像
input_image_path1 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl1_1024/1024_model_output.tiff'
input_image_path2 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl2_1024/1024_model_output.tiff'
input_image_path3 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl6_1024/1024_model_output.tiff'  # 3つ目の画像
# input_image_path3 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w60_1024/1024_model_output.tiff'  # 3つ目の画像
# input_image_path3 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl4_1024/1024_model_output.tiff'  # 3つ目の画像
# input_image_path4 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w60_1024/1024_model_output.tiff'  # 4つ目の画像
input_image_path4 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7_1024/1024_model_output.tiff'
# input_image_path4 = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_512x512_no_resize_1024x1024.tiff'  # 4つ目の画像
# input_image_path4 = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_uw30/trained_siren_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_1024/1024_model_output.tiff'  # 4つ目の画像
pixel = 128  # 変更

# 画像を開く
image1 = Image.open(input_image_path1)
image2 = Image.open(input_image_path2)
image3 = Image.open(input_image_path3)
image4 = Image.open(input_image_path4)

# 画像をグレースケールに変換
image_gray1 = image1.convert("L")
image_gray2 = image2.convert("L")
image_gray3 = image3.convert("L")
image_gray4 = image4.convert("L")

# Numpy配列に変換
image_np1 = np.array(image_gray1)
image_np2 = np.array(image_gray2)
image_np3 = np.array(image_gray3)
image_np4 = np.array(image_gray4)

# 画像サイズの比率を計算
scaling_factor1 = image_np2.shape[0] / image_np1.shape[0]
scaling_factor2 = image_np3.shape[0] / image_np1.shape[0]
scaling_factor3 = image_np4.shape[0] / image_np1.shape[0]

# 画像表示用のセットアップ（横に4つの画像を並べて表示）
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
ax1.imshow(image_np1, cmap='gray')
ax1.set_title('Image 1')
ax2.imshow(image_np2, cmap='gray')
ax2.set_title('Image 2')
ax3.imshow(image_np3, cmap='gray')
ax3.set_title('Image 3')
ax4.imshow(image_np4, cmap='gray')
ax4.set_title('Image 4')

# 各画像用の赤い縦線プロット
line_plot1, = ax1.plot([], [], 'r', linewidth=1.5)
line_plot2, = ax2.plot([], [], 'r', linewidth=1.5)
line_plot3, = ax3.plot([], [], 'r', linewidth=1.5)
line_plot4, = ax4.plot([], [], 'r', linewidth=1.5)

# 輝度グラフのセットアップ
fig_brightness, ax_brightness = plt.subplots()
line_brightness1, = ax_brightness.plot([], [], label="Image 1")
line_brightness2, = ax_brightness.plot([], [], label="Image 2")
line_brightness3, = ax_brightness.plot([], [], label="Image 3")
line_brightness4, = ax_brightness.plot([], [], label="Image 4")
ax_brightness.set_xlabel("Column (Pixel)")
ax_brightness.set_ylabel("Brightness (0-255)")
ax_brightness.grid(True)
ax_brightness.legend()

# 赤い縦線を更新する関数
def update_red_lines(col):
    line_plot1.set_data([col, col], [0, image_np1.shape[0]])  # Image 1 に赤線
    line_plot2.set_data([col * scaling_factor1, col * scaling_factor1], [0, image_np2.shape[0]])  # Image 2 に赤線
    line_plot3.set_data([col * scaling_factor2, col * scaling_factor2], [0, image_np3.shape[0]])  # Image 3 に赤線
    line_plot4.set_data([col * scaling_factor3, col * scaling_factor3], [0, image_np4.shape[0]])  # Image 4 に赤線
    fig.canvas.draw()  # 描画を更新

# 輝度を縦方向に更新する関数
def update_brightness_vertical(row, col, image_np, pixel, scaling_factor=1):
    row = int(row * scaling_factor)
    col = int(col * scaling_factor)
    pixel = int(pixel * scaling_factor)

    if 0 <= row < image_np.shape[0]:
        start_col = max(0, col - pixel // 2)
        end_col = min(image_np.shape[1], start_col + pixel)
        start_row = max(0, row - pixel // 2)
        end_row = min(image_np.shape[0], start_row + pixel)

        brightness_values = image_np[start_row:end_row, col]
        cropped_image = image_np[start_row:end_row, start_col:end_col]
        return brightness_values, cropped_image

# 縦方向の輝度グラフを更新する関数
def update_brightness_graph_vertical(row, col):
    # 画像1の縦方向の輝度値を取得
    brightness_values1, cropped_image1 = update_brightness_vertical(row, col, image_np1, pixel)
    brightness_values2, cropped_image2 = update_brightness_vertical(row, col, image_np2, pixel, scaling_factor1)
    # brightness_values3, cropped_image3 = update_brightness_vertical(row, col, processed_image3, pixel, scaling_factor2)
    brightness_values3, cropped_image3 = update_brightness_vertical(row, col, image_np3, pixel, scaling_factor2)
    brightness_values4, cropped_image4 = update_brightness_vertical(row, col, image_np4, pixel, scaling_factor3)
    
    # ピクセル数が小さい方のデータをスケーリングして、輝度値を同じ長さに合わせる
    max_length = max(len(brightness_values1), len(brightness_values2), len(brightness_values3))
    if len(brightness_values1) < max_length:
        brightness_values1 = zoom(brightness_values1, max_length / len(brightness_values1))  # 1つ目の輝度値を拡張
    if len(brightness_values2) < max_length:
        brightness_values2 = zoom(brightness_values2, max_length / len(brightness_values2))  # 2つ目の輝度値を拡張
    if len(brightness_values3) < max_length:
        brightness_values3 = zoom(brightness_values3, max_length / len(brightness_values3))  # 3つ目の輝度値を拡張
    if len(brightness_values4) < max_length:
        brightness_values4 = zoom(brightness_values4, max_length / len(brightness_values4))


    # グラフを更新
    line_brightness1.set_data(np.arange(len(brightness_values1)), brightness_values1)
    line_brightness2.set_data(np.arange(len(brightness_values2)), brightness_values2)
    line_brightness3.set_data(np.arange(len(brightness_values3)), brightness_values3)
    line_brightness4.set_data(np.arange(len(brightness_values4)), brightness_values4)
    
    ax_brightness.set_xlim(0, max(len(brightness_values1), len(brightness_values2), len(brightness_values3), len(brightness_values4)))
    ax_brightness.set_ylim(0, 255)
    ax_brightness.set_title(f"Row {row} Col {col} Brightness Comparison")
    fig_brightness.canvas.draw()
    
    # 抽出された部分を並べて表示
    fig_crop, (ax_crop1, ax_crop2, ax_crop3, ax_crop4) = plt.subplots(1, 4, figsize=(16, 4))  # 横に並べて表示
    ax_crop1.imshow(cropped_image1, cmap='gray')
    ax_crop1.set_title(f"Image 1 - Row {row}, Col {col}")
    # ax_crop1.axvline(x=pixel//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop1.axvline(x=44, color='red', linestyle='-', linewidth=1.5)
    
    ax_crop2.imshow(cropped_image2, cmap='gray')
    ax_crop2.set_title(f"Image 2 - Row {int(row*scaling_factor1)}, Col {int(col*scaling_factor1)}")
    # ax_crop2.axvline(x=pixel*scaling_factor1//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop2.axvline(x=44*scaling_factor1, color='red', linestyle='-', linewidth=1.5)  
    
    ax_crop3.imshow(cropped_image3, cmap='gray')
    ax_crop3.set_title(f"Image 3 - Row {int(row*scaling_factor2)}, Col {int(col*scaling_factor2)}")
    # ax_crop3.axvline(x=pixel*scaling_factor2//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop3.axvline(x=44*scaling_factor2, color='red', linestyle='-', linewidth=1.5) 
    
    ax_crop4.imshow(cropped_image4, cmap='gray')
    ax_crop4.set_title(f"Image 4 - Row {int(row*scaling_factor3)}, Col {int(col*scaling_factor3)}")
    # ax_crop4.axvline(x=pixel*scaling_factor2//2, color='red', linestyle='-', linewidth=1.5)  # 中心に赤線を引く
    ax_crop4.axvline(x=44*scaling_factor3, color='red', linestyle='-', linewidth=1.5) 
    
    plt.show()

# クリックイベントで縦軸のグラフを表示する
def on_click_vertical(event):
    if event.inaxes in [ax1, ax2, ax3, ax4]:
        row = int(event.ydata)
        col = int(event.xdata)
        update_brightness_graph_vertical(row, col)
        update_red_lines(col)
        print(f"クリックされた座標: 行={row}, 列={col}")

# テキストボックスのセットアップ
fig_textbox, (ax_textbox_row, ax_textbox_col) = plt.subplots(1, 2, figsize=(8, 1))
text_box_row = TextBox(ax_textbox_row, "Row:", initial="0")
text_box_col = TextBox(ax_textbox_col, "Col:", initial="0")

# Pushボタンを追加
ax_button_vertical = plt.axes([0.8, 0.05, 0.1, 0.075])
button_vertical = Button(ax_button_vertical, 'Push Vertical')

# 座標を手動で指定して更新する関数
def submit_coordinates_vertical(event):
    try:
        row = int(text_box_row.text)
        col = int(text_box_col.text)
        update_brightness_graph_vertical(row, col)
        update_red_lines(col)
    except ValueError:
        print("有効な行番号と列番号を入力してください")

button_vertical.on_clicked(submit_coordinates_vertical)

# マウスクリックイベントを登録
fig.canvas.mpl_connect('button_press_event', on_click_vertical)

plt.show()
