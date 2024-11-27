import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict

import time

def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords, save_output=False, save_dir="layer_outputs"): # def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        # output = self.net(coords)
        # return output, coords
        intermediate_outputs = []  # 中間出力を格納するリスト # なくていい（将来的に各層の出力を保存して解析したい場合やデバッグのために出力を確認したい場合には、この部分が役立つことがある）
        
        x = coords
        for i, layer in enumerate(self.net):
            x = layer(x)  # レイヤーを通して出力を計算
            intermediate_outputs.append(x)

            # 各層の出力を保存する処理
            if save_output:
                self.save_layer_output(x, i + 1, save_dir)
                
        return x, coords

    def save_layer_output(self, layer_output, layer_index, save_dir):
        # 出力テンソルを [batch_size, num_pixels, num_channels] の形で保存する
        output_dir = os.path.join(save_dir, f"/mnt/siren/explore_siren/1017/layer_{layer_index}") #ここ変える
        os.makedirs(output_dir, exist_ok=True)

        # 例えば、torch.Size([1, 65536, 256]) の形状である場合、65536 は 256x256 ピクセル
        batch_size, num_pixels, num_channels = layer_output.shape
        image_size = int(np.sqrt(num_pixels))  # 画像サイズ (例: 256x256)

        # 各チャンネルごとに画像を保存
        for channel in range(num_channels):
            channel_output = layer_output[0, :, channel].detach().cpu().numpy().reshape(image_size, image_size)

            # 画像を保存するだけで表示はしない
            plt.imsave(os.path.join(output_dir, f'channel_{channel + 1}.png'), channel_output, cmap='gray')
            
    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1
        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_onsen_tensor(sidelength):
    # img = Image.open('/mnt/siren/data_img_yasuda/line3.png').convert('L').resize((512, 512))
    img = Image.open('/mnt/siren/data_img_yasuda/cat.png').convert('L').resize((512, 512))
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_onsen_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageFittingデータセットのインスタンスを作成
cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

# Sirenモデルのインスタンスを作成
img_siren = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device) # default(hidden_features): 256

# 訓練の設定
total_steps = 10000#500
steps_til_summary = 200#10
output_dir = '/mnt/siren/explore_siren/1017'# 変更
os.makedirs(output_dir, exist_ok=True)

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.to(device), ground_truth.to(device)

for step in range(total_steps):
    # 最後のステップかどうかを判定  #step == total_steps - 1 の結果は True か False になる
    save_output = step == total_steps - 1
    
    # 最後のステップでのみ save_output=True で forward を実行
    model_output, coords = img_siren(model_input, save_output=save_output, save_dir=f"step_{step}_outputs")

    loss = ((model_output - ground_truth)**2).mean()
    
    # 訓練の途中で進捗を表示（定期的に実行）
    if not step % steps_til_summary:
        print(f"Step {step}, Total loss {loss.item():0.6f}")
        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy(), cmap='gray')
        axes[0].set_title('Model Output')
        axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy(), cmap='gray')
        axes[1].set_title('Gradient Norm')
        axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy(), cmap='gray')
        axes[2].set_title('Laplacian')
        
        plt.savefig(os.path.join(output_dir, f'step_{step}.png'))
        plt.close(fig)

    # 誤差逆伝播とパラメータの更新
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # 最後のステップで画像保存メッセージ
    if save_output:
        print(f"Saving images for the final step: {step}")