import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import math  # 切り上げのために使用

# 1024ピクセル画像内での開始座標（例: x=100, y=200）
start_x, start_y = 212, 668 # 変更箇所

# 入力画像のパス
image_path = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff'# 変更箇所

# 出力画像のパス
resized_image_path = f'/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_64x64_no_resize_{start_x}_{start_y}.tiff'# 変更箇所



# 出力ディレクトリの作成
output_dir = os.path.dirname(resized_image_path)
os.makedirs(output_dir, exist_ok=True)

# 画像をロード
image = Image.open(image_path)

# 画像をnumpy配列に変換
image_array = np.array(image)

# 入力画像サイズの確認
image_height, image_width = image_array.shape[:2]
if image_width < 1024 or image_height < 1024:
    raise ValueError("The input image must be at least 1024x1024 pixels.")
if start_x + 128 > 1024 or start_y + 128 > 1024:
    raise ValueError("The specified 128x128 region exceeds the 1024x1024 boundaries.")

# 1024ピクセル画像全体で奇数行と奇数列を抽出（これにより512ピクセル画像が生成される）
odd_large_region = image_array[0:1024:2, 0:1024:2]

# 切り上げ処理を適用して128ピクセル領域を抽出
sub_start_y = math.ceil(start_y / 2)
sub_start_x = math.ceil(start_x / 2)
sub_end_y = math.ceil((start_y + 128) / 2)
sub_end_x = math.ceil((start_x + 128) / 2)

# 128ピクセル領域を抽出（結果的に64x64ピクセル）
sub_region = odd_large_region[sub_start_y:sub_end_y, sub_start_x:sub_end_x]

# データ型を明示的に変換（画像形式に互換性を持たせる）
sub_region = sub_region.astype(np.uint8)

# 抽出した配列を画像に変換
resized_image = Image.fromarray(sub_region)

# 保存
resized_image.save(resized_image_path)

saved_image_size = resized_image.size
print(f"Resized image saved to: {resized_image_path}")
print(f"Saved image size: {saved_image_size[0]}x{saved_image_size[1]} pixels")  # サイズを表示


# PIL画像からの表示
plt.imshow(resized_image, cmap='gray')  # グレースケール画像として表示
plt.title("Extracted 64x64 Region")     # タイトルを追加
plt.axis('off')                         # 軸を非表示にする
plt.show()                              # 画像を表示
