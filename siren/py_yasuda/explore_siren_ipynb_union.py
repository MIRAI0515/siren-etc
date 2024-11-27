from PIL import Image
import os

def concatenate_images_vertically(image_dir, output_path):
    images = []
    
    # ディレクトリ内のファイルをすべて取得し、'channel_'で始まるファイルをフィルタリング
    image_files = sorted([f for f in os.listdir(image_dir) if f.startswith('channel_') and f.endswith('.png')])
    num_images = len(image_files)  # 画像の数を取得
    
    if num_images == 0:
        print(f"No images found in {image_dir}")
        return
    
    # 画像を読み込む
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)
        images.append(img)
    
    # 各画像のサイズを取得 (ここでは全ての画像が同じサイズであると仮定)
    width, height = images[0].size
    
    # 合計の高さは、画像の高さ × 画像の数
    total_height = height * num_images
    
    # 結合された画像用の空のキャンバスを作成
    concatenated_image = Image.new('RGB', (width, total_height))
    
    # 各画像をキャンバスに順番に貼り付ける
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += height  # Y軸方向にオフセットをずらす
    
    # 結合された画像を保存
    concatenated_image.save(output_path)
    print(f"Images successfully concatenated and saved to {output_path}")
    
# 画像が保存されているディレクトリと出力ファイルのパス
image_directory = "/mnt/siren/explore_siren/0926_hidden50_240613_2204_EGFP/layer_4"  # 各チャンネルの画像が保存されているフォルダ
output_file = "/mnt/siren/explore_siren/0926_hidden50_240613_2204_EGFP/layer_4/union.png"  # 出力ファイルのパス

# 画像を縦に結合
concatenate_images_vertically(image_directory, output_file)
