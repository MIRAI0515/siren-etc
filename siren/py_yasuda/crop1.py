from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.widgets import TextBox, Button

# 画像のパスを指定
input_image_path = '/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7_1024/1024_model_output.tiff'  # 変更箇所
n = 8  # 切り出し領域の分割数

# 画像を開く
image = Image.open(input_image_path)
image_np = np.array(image)  # PIL Imageをnumpy配列に変換
print("画像のサイズ:", image_np.shape)

# グローバル変数
rect = None
start_x, start_y = 0, 0  # 初期座標
width, height = 1024 // n, 1024 // n  # 切り取り領域の幅と高さ

# 左クリックでプレビュー（切り取る領域の赤い枠を表示）
def on_left_click(event):
    global rect, start_x, start_y
    if event.button == 1 and event.inaxes is not None:  # 左クリック
        start_x, start_y = int(event.xdata), int(event.ydata)  # クリックした座標を取得

        # 赤い矩形がすでにある場合は削除
        if rect:
            rect.remove()

        # 赤い矩形を描画
        rect = patches.Rectangle((start_x, start_y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()

# 右クリックで切り取り確定
def on_right_click(event):
    if event.button == 3 and event.inaxes is not None:  # 右クリック
        save_cropped_image(start_x, start_y)

# 切り取りと保存の処理
def save_cropped_image(start_x, start_y):
    # 指定された位置から領域を切り取る
    cropped_image = image.crop((start_x, start_y, start_x + width, start_y + height))
    output_image_path = f'/mnt/siren/mip_nd2_yasuda/240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7_1024_{start_x}_{start_y}_{width}.tiff'  # 変更箇所
    cropped_image.save(output_image_path)
    print(f"画像を保存しました: {output_image_path}")
    
    # 切り取った画像を表示
    cropped_image.show()

# テキストボックスから座標を取得して切り取りを実行
def submit_coordinates(event):
    try:
        x = int(text_box_x.text)
        y = int(text_box_y.text)
        save_cropped_image(x, y)
    except ValueError:
        print("有効な座標を入力してください")

# 画像を表示してクリックイベントを設定
fig, ax = plt.subplots()
ax.imshow(image_np, cmap='gray')

# 左クリックでプレビュー
fig.canvas.mpl_connect('button_press_event', on_left_click)
# 右クリックで切り取り確定
fig.canvas.mpl_connect('button_press_event', on_right_click)

# テキストボックスとボタンを追加
fig_textbox, (ax_textbox_x, ax_textbox_y) = plt.subplots(1, 2, figsize=(8, 1))
text_box_x = TextBox(ax_textbox_x, "X:", initial="0")
text_box_y = TextBox(ax_textbox_y, "Y:", initial="0")

# ボタンを追加
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Push')

# ボタンのクリックで座標から切り取りを実行
button.on_clicked(submit_coordinates)

plt.show()
