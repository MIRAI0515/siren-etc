import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from collections import OrderedDict
import re

HIDDEN_LAYER = 7 # 変更箇所
OMEGA = 30 # 変更箇所

# Define Sine activation layer
class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(OMEGA * input)

# Define FCBlock class
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=True):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(in_features, hidden_features), Sine())]
        for _ in range(num_hidden_layers):
            layers.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), Sine()))
        if outermost_linear:
            layers.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            layers.append(nn.Sequential(nn.Linear(hidden_features, out_features), Sine()))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Define SingleBVPNet model
class SingleBVPNet(nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_features=256, num_hidden_layers=HIDDEN_LAYER):
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=True)

    def forward(self, coords):
        return self.net(coords)
    
# 輝度値統計情報の計算関数
def calculate_statistics(data):
    flat_data = data.flatten()  # 2次元配列を1次元に変換
    min_value = np.min(flat_data)
    max_value = np.max(flat_data)
    avg_value = np.mean(flat_data)
    median_value = np.median(flat_data)

    return {
        'min': min_value,
        'max': max_value,
        'average': avg_value,
        'median': median_value
    }
    
# 輝度値修正の関数（テンソル版）
def process_tensor(data_tensor):
    rows, cols = data_tensor.shape
    processed_tensor = data_tensor.clone()

    # テンソルの最小値を取得
    min_value = torch.min(data_tensor).item()
    correction_value = min_value - 0.001  # 修正値を最小値から-0.001とする

    # マトリックス内の全要素に対して、周囲と比較（境界は無視）
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 上下左右の要素よりも小さい場合、その値を条件に応じて修正
            if (data_tensor[i, j] < data_tensor[i - 1, j] and  # 上
                data_tensor[i, j] < data_tensor[i + 1, j] and  # 下
                data_tensor[i, j] < data_tensor[i, j - 1] and  # 左
                data_tensor[i, j] < data_tensor[i, j + 1]):    # 右
                processed_tensor[i, j] = correction_value  # 修正された値を設定
                
    return processed_tensor, correction_value

# Load pre-trained model
model = SingleBVPNet(in_features=2, out_features=1, hidden_features=256, num_hidden_layers=HIDDEN_LAYER) 
model_path = '/mnt/siren/logs/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7/checkpoints/model_final.pth' # 変更箇所
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

# Generate a grid of coordinates as input
def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid.reshape(-1, dim)

# Set up the input and get model output
sidelen = 1024  # 変更箇所 #入力したい座標の数
output_dir = f'/mnt/siren/explore_siren/experiment_240613_2204_EGFP_resized_512x512_no_resize_sine_10000_w30_hl7_{sidelen}' # 変更箇所
os.makedirs(output_dir, exist_ok=True)
model_input = get_mgrid(sidelen).float()
model_output = model(model_input)#.detach().cpu().numpy().reshape(sidelen, sidelen)
model_output_np = model_output.detach().cpu().numpy().reshape(sidelen, sidelen) 
# 出力サイズを保存
output_size = sidelen 

# 輝度値の統計情報を計算
stats = calculate_statistics(model_output_np)
# 統計情報の表示
print("Model Output")
print(f"Min: {stats['min']}")
print(f"Max: {stats['max']}")
print(f"Average: {stats['average']}")
print(f"Median: {stats['median']}")


# -1より小さいものは-1に、1より大きいものは1に設定（torch.clampを使用）
model_output_clamp = torch.clamp(model_output, min=-1, max=1)
# -1~1を0~255に変換
model_output_clamp = ((model_output_clamp + 1) * 127.5).to(torch.uint8) # 整数でないと、model_output_processed_rgb.pngが出力されなかった。

# 出力をCPUに移してNumPy配列に変換
model_output_clamp_np = model_output_clamp.cpu().view(sidelen, sidelen).detach().numpy()

# （SIRENを通した後の）画像を保存（輝度値0~255を考慮）
image_file2 = os.path.join(output_dir, f'{output_size}_model_output.png')
plt.imshow(model_output_clamp_np, cmap='gray')
plt.title(f'Model Output ({output_size}x{output_size})')
plt.colorbar()
plt.savefig(image_file2)
plt.close()
print(f"モデルの出力を画像として '{image_file2}' に保存しました。")


# NumPy配列をTIFF形式で保存する
image_file2_tiff = os.path.join(output_dir, f'{output_size}_model_output.tiff')
# PILを使用してTIFF画像を保存
image = Image.fromarray(model_output_clamp_np)
image.save(image_file2_tiff, format="TIFF")
print(f"モデルの出力をTIFF画像として '{image_file2_tiff}' に保存しました。")


# （SIRENを通した後の）テキストファイルを保存（輝度値0~255を考慮）
model_output_txt = os.path.join(output_dir, 'model_output.txt')
np.savetxt(model_output_txt, model_output_clamp_np, fmt='%d') # model_output_clamp_npではなくmodel_output_npにしている。
print(f"モデルの出力をテキストとして '{model_output_txt}' に保存しました。")

# 輝度値の統計情報を計算
stats1 = calculate_statistics(model_output_clamp_np)
# 統計情報の表示
print("Model Output")
print(f"Min: {stats1['min']}")
print(f"Max: {stats1['max']}")
print(f"Average: {stats1['average']}")
print(f"Median: {stats1['median']}")

# データのサイズが2×2以上の場合に処理を行う
if output_size >= 3 and output_size >= 3:
    # データをテンソルとして処理（輝度値修正を適用した後）
    processed_data, modified_value = process_tensor(model_output.view(output_size, output_size))
   # 結果を保存するためにテンソルをNumPy配列に変換
    processed_data_np = processed_data.cpu().detach().numpy()
    
    # 1. 修正された部分のマスクを保存
    # modified_value には（最小値 - 0.001）が格納されている
    modified_mask = (processed_data.cpu().detach().numpy() == modified_value)

    # 2. -1より小さいものは-1に、1より大きいものは1に設定
    processed_data_clamp_np = np.clip(processed_data_np, -1, 1)

    # 3. -1~1を0~255に変換
    processed_data_clamp_np = ((processed_data_clamp_np + 1) * 127.5).astype(np.uint8) # 整数でないと、model_output_processed_rgb.pngが出力されなかった。
      
    # 4. グレースケール行列をRGBに複製
    rgb_image = np.stack([processed_data_clamp_np,processed_data_clamp_np, processed_data_clamp_np], axis=-1)

    # 5. 修正された部分を青チャンネル(B)で強調
    rgb_image[modified_mask, 2] = 0  # Bチャンネルを0にして青色を強調
    rgb_image[modified_mask, 0] = 255  # Rチャンネルを強調して黄色を表現
    rgb_image[modified_mask, 1] = 255  # Gチャンネルも強調

    # （輝度値が修正された箇所をわかりやすくした）画像を保存
    image_file3 = os.path.join(output_dir, f'{sidelen}_model_output_processed_rgb.png')
    plt.imshow(rgb_image)
    plt.title('Model Output with Highlighted Modifications')
    plt.savefig(image_file3)
    plt.close()
    print(f"画像が保存されました: {image_file3}")
else:
    print("データが小さすぎて処理できません（行列サイズが2×2以下です）。")

# （輝度値修正後の）画像を保存（輝度値0~255を考慮）
image_file5 = os.path.join(output_dir, f'{output_size}_model_output_processed.png')
plt.imshow(processed_data_clamp_np, cmap='gray')
plt.title(f'Model Output Brightness({output_size}x{output_size})')
plt.colorbar()
plt.savefig(image_file5)
plt.close()
print(f"モデルの出力を画像として '{image_file5}' に保存しました。")

# NumPy配列をTIFF形式で保存する
image_file5_tiff = os.path.join(output_dir, f'{output_size}_model_output_processed.tiff')
# PILを使用してTIFF画像を保存
image = Image.fromarray(processed_data_clamp_np)
image.save(image_file5_tiff, format="TIFF")
print(f"モデルの出力をTIFF画像として '{image_file5_tiff}' に保存しました。")

# （輝度値修正後の）テキストファイルを保存（輝度値0~255を考慮）
model_output_processed_txt = os.path.join(output_dir, 'model_output_processed.txt')
np.savetxt(model_output_processed_txt, processed_data_clamp_np, fmt='%d')
print(f"モデルの出力をテキストとして '{model_output_processed_txt}' に保存しました。")

# 輝度値の統計情報を計算
stats2 = calculate_statistics(processed_data_clamp_np)
# 統計情報の表示
print("Model Output（輝度値修正後）")
print(f"Min: {stats2['min']}")
print(f"Max: {stats2['max']}")
print(f"Average: {stats2['average']}")
print(f"Median: {stats2['median']}")