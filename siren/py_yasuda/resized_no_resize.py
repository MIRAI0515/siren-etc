from PIL import Image
import numpy as np

# 画像をロード
image_path = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff'
image = Image.open(image_path)

# 画像をnumpy配列に変換
image_array = np.array(image)

# 奇数行と奇数列のみを抽出する（インデックス0から始まるため、::2を使用）
# まずは512×512の範囲まで奇数行列を抽出する
odd_rows_and_columns_image_array = image_array[0:1024:2, 0:1024:2]

# 抽出した配列を画像に変換
resized_image = Image.fromarray(odd_rows_and_columns_image_array)

# リサイズ不要で512x512になるので、そのまま保存
resized_image_path = '/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_512x512_no_resize.tiff'
resized_image.save(resized_image_path)

print(f"Resized image saved to: {resized_image_path}")