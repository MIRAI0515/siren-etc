import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 画像の読み込み（グレースケール画像として読み込む）
# Original  = cv2.imread('/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP.tiff', cv2.IMREAD_GRAYSCALE)
Original  = cv2.imread('/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_128_235_831.tiff', cv2.IMREAD_GRAYSCALE) # 変更
Distorted = cv2.imread('/mnt/siren/mip_nd2_yasuda/240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl6_1024_235_831_128.tiff', cv2.IMREAD_GRAYSCALE) # 変更
# Distorted = cv2.imread('/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7_1024/1024_model_output.tiff', cv2.IMREAD_GRAYSCALE) # 変更
# Distorted = cv2.imread('/mnt/siren/mip_nd2_yasuda/Ex03_No.14_PBS_SHG_ELN_FBN_DAPI_880nm003_MIP/240613_2204_EGFP_resized_512x512_no_resize.tiff', cv2.IMREAD_GRAYSCALE) # エラー

# 画素値の読み込み
pixel_value_Ori = Original.flatten().astype(float)
pixel_value_Dis = Distorted.flatten().astype(float)

# 画像サイズを取得
imageHeight, imageWidth = Original.shape

# 画素数
N = imageHeight * imageWidth 

# MSE（平均二乗誤差）を計算
MSE = np.mean((pixel_value_Ori - pixel_value_Dis) ** 2)

# PSNRを計算
if MSE == 0:  # 完全に同じ画像の場合
    PSNR = float('inf')
else:
    PSNR = 10 * math.log10(255 * 255 / MSE)

print('PSNR:', PSNR)

# 画像の表示
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(Original, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(Distorted, cmap='gray')
axs[1].set_title('Distorted Image')
axs[1].axis('off')

plt.show()
