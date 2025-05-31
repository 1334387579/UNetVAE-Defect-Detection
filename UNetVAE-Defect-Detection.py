# %%capture --no-stderr
# 以下是一条Jupyter魔法命令，用于抑制此单元格的stderr输出。
!pip install ultralytics -q 
!pip install tqdm opencv-python-headless pyyaml scikit-learn matplotlib numpy torch torchvision Pillow -q

import os
from pathlib import Path
import random
import shutil
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils 

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# 解决Matplotlib中文显示问题 (尝试，如果环境中有这些字体)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False  
    print("Matplotlib中文字体已尝试设置。")
except Exception as e_font:
    print(f"设置Matplotlib中文字体失败: {e_font}。图表中的中文可能无法正确显示。")


print("阶段 0: 配置与设置 - 基于U-Net VAE的缺陷检测 (修正解码器输出尺寸)")

# --- 数据集配置 ---
BASE_TEST_DATA_PATH = Path("/kaggle/input/wood-vae/wood/test")
TRAIN_DATA_NORMAL_PATH = Path("/kaggle/input/wood-vae/wood/train/good")

TEST_DATA_PATHS = {
    "color": BASE_TEST_DATA_PATH / "color",
    "combined": BASE_TEST_DATA_PATH / "combined",
    "hole": BASE_TEST_DATA_PATH / "hole",
    "good_test": BASE_TEST_DATA_PATH / "good" ,
    "liquid": BASE_TEST_DATA_PATH / "liquid",
    "scratch": BASE_TEST_DATA_PATH / "scratch"
}

# --- 输出配置 ---
OUTPUT_DIR = Path("/kaggle/working/vae_wood_defect_output_img1024_unet_fixed_decoder") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "vae_wood_detector_img1024_unet_fixed_decoder.pth"
RESULTS_DIR = OUTPUT_DIR / "results_img1024_unet_fixed_decoder"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 模型与图像配置 ---
IMG_SIZE = (1024, 1024)
LATENT_DIM = 256       
CHANNELS = 3          

# --- 训练配置 ---
EPOCHS = 360            # !!! 临时减少EPOCHS用于测试运行，实际训练需改大 !!!
BATCH_SIZE = 16         # !!! 对于1024x1024 和 U-Net VAE，BATCH_SIZE必须为1 !!!
LEARNING_RATE = 1e-4 
KLD_WEIGHT = 0.0001    

# --- 设备配置 --- (与之前相同)
if torch.cuda.is_available():
    num_gpus_available = torch.cuda.device_count()
    print(f"检测到 {num_gpus_available} 个可用的GPU。")
    if num_gpus_available > 0: 
        train_main_device = torch.device(f"cuda:{0}") 
        if num_gpus_available > 1:
            train_device_ids_for_dp = [i for i in range(num_gpus_available)]
            print(f"训练将使用DataParallel，在GPUs: {train_device_ids_for_dp} 上运行。主设备: {train_main_device}。")
        else: 
            train_device_ids_for_dp = None 
            print(f"训练将在单个GPU: {train_main_device} 上运行。")
    else: 
        train_main_device = torch.device("cpu")
        train_device_ids_for_dp = None
        print("警告: CUDA可用但未检测到GPU？回退到CPU进行训练。")
else: 
    train_main_device = torch.device("cpu")
    train_device_ids_for_dp = None
    print("未检测到CUDA，将使用CPU进行训练。")
eval_device = train_main_device 
print(f"训练主设备: {train_main_device}")
print(f"DataParallel设备ID列表 (若使用): {train_device_ids_for_dp}")
print(f"评估/推理设备: {eval_device}")
if eval_device.type == 'cuda':
     print(f"评估/推理 GPU名称: {torch.cuda.get_device_name(eval_device.index if eval_device.index is not None else 0)}")

# -----------------------------------------------------------------------------
# 1. 数据加载与预处理 (与之前相同)
# -----------------------------------------------------------------------------
print("\n阶段 1: 数据加载与预处理")
data_transforms = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])
class WoodDataset(Dataset):
    def __init__(self, root_dir_list, transform=None, is_normal=True):
        self.transform = transform; self.image_paths = []; self.labels = [] 
        if not isinstance(root_dir_list, list): root_dir_list = [root_dir_list]
        for root_dir_path in root_dir_list:
            root_dir = Path(root_dir_path)
            if not root_dir.exists(): print(f"警告: 目录 {root_dir} 不存在。"); continue
            current_paths = [p for ext in ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"] for p in root_dir.glob(ext)]
            self.image_paths.extend(current_paths)
            self.labels.extend([0 if is_normal else 1] * len(current_paths))
        print(f"从 {root_dir_list} 中找到 {len(self.image_paths)} 张图像。")
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx], self.labels[idx]
        try: 
             image = Image.open(img_path).convert("RGB") 
             if self.transform: image = self.transform(image)
             return image, label, str(img_path) 
        except Exception as e: 
             print(f"错误: 加载图像 {img_path}: {e}")
             return None, label, str(img_path)
def custom_collate_fn_vae(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None, None, None
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return images, labels, paths
train_dataset_normal = WoodDataset(TRAIN_DATA_NORMAL_PATH, transform=data_transforms, is_normal=True)
if not train_dataset_normal or len(train_dataset_normal) == 0: print(f"严重错误: 训练集 '{TRAIN_DATA_NORMAL_PATH}' 为空。退出。"); exit()
train_loader_normal = DataLoader(train_dataset_normal, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn_vae, pin_memory=train_main_device.type=='cuda')
all_test_img_list, all_test_label_list = [], []
good_test_path = TEST_DATA_PATHS.get("good_test")
if good_test_path and good_test_path.exists():
    paths = [p for ext in ["*.png","*.jpg","*.jpeg"] for p in good_test_path.glob(ext)]
    all_test_img_list.extend(paths); all_test_label_list.extend([0]*len(paths))
for defect_type, path in TEST_DATA_PATHS.items():
    if defect_type != "good_test" and path.exists():
        paths = [p for ext in ["*.png","*.jpg","*.jpeg"] for p in path.glob(ext)]
        all_test_img_list.extend(paths); all_test_label_list.extend([1]*len(paths))
class CombinedTestDataset(Dataset):
    def __init__(self, image_p_list, label_list, transform=None): 
        self.image_paths = image_p_list; self.labels = label_list; self.transform = transform 
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx], self.labels[idx]
        try: 
             image = Image.open(img_path).convert("RGB") 
             if self.transform: image = self.transform(image)
             return image, label, str(img_path)
        except Exception as e_img_load: 
             print(f"错误: 加载测试图像 {img_path}: {e_img_load}")
             return None, label, str(img_path)
test_dataset_obj = CombinedTestDataset(all_test_img_list, all_test_label_list, transform=data_transforms) if all_test_img_list else None
test_loader = DataLoader(test_dataset_obj, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn_vae, pin_memory=eval_device.type=='cuda') if test_dataset_obj and len(test_dataset_obj) > 0 else None
if test_dataset_obj and len(test_dataset_obj) > 0: print(f"测试数据集已加载 ({len(test_dataset_obj)} 张图像)。")
else: print("警告: 未能加载任何测试图像或测试数据集为空。")

# -----------------------------------------------------------------------------
# 2. VAE 模型定义 (U-Net Like VAE for 1024x1024, 修正解码器)
# -----------------------------------------------------------------------------
print("\n阶段 2: 定义U-Net风格的VAE模型 (img_size=1024, 修正解码器)")

class UNetVAE(nn.Module):
    def __init__(self, channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE[0]):
        super(UNetVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        # 编码器通道数
        self.enc_channels_spec = [channels, 32, 64, 128, 256, 512, 512, 512, 512] 

        self.encoder_blocks = nn.ModuleList()
        current_channels = self.enc_channels_spec[0]
        for i in range(len(self.enc_channels_spec) - 1):
            out_ch = self.enc_channels_spec[i+1]
            self.encoder_blocks.append(
                self._conv_block(current_channels, out_ch, 
                                 no_bn_relu_for_last_conv=(i == len(self.enc_channels_spec) - 2) 
                                )
            )
            current_channels = out_ch
        
        self.final_feature_map_size = img_size // (2**len(self.encoder_blocks)) 
        fc_input_size = self.enc_channels_spec[-1] * self.final_feature_map_size**2 
        print(f"UNetVAE DEBUG: img_size={self.img_size}, final_feat_map_size={self.final_feature_map_size}, fc_input_size={fc_input_size}")

        self.fc_mu = nn.Linear(fc_input_size, latent_dim)
        self.fc_logvar = nn.Linear(fc_input_size, latent_dim)
        
        self.decoder_input_fc = nn.Linear(latent_dim, fc_input_size)
        
        self.decoder_upconv_blocks = nn.ModuleList()
        self.decoder_conv_blocks = nn.ModuleList()
        
        # 解码器通道数逻辑
        # enc_channels_spec: [3, 32, 64, 128, 256, 512, 512, 512, 512]
        # 瓶颈层输出通道数 (也是解码器第一个上采样块的输入通道数): self.enc_channels_spec[-1] = 512
        # 跳跃连接的通道数 (从编码器的倒数第二层特征图开始，到第一层特征图):
        # skip_connection_channels_list: [512, 512, 512, 256, 128, 64, 32] (对应 enc_c7 ... enc_c1)
        
        current_dec_channels = self.enc_channels_spec[-1] # 512 (来自瓶颈层之后的fc_input_size reshape)
        
        # 解码器需要 len(self.encoder_blocks) - 1 个 (upconv + conv_after_skip) 块
        # 再加上最后一步上采样到全尺寸
        num_decoder_stages = len(self.encoder_blocks) -1 # 7个阶段的跳跃连接

        for i in range(num_decoder_stages): # i from 0 to 6
            skip_channel_idx = -(i + 2) # 从 enc_channels_spec 取跳跃连接的通道数: -2, -3, ..., -8
                                        # self.enc_channels_spec[-2] 是 enc_c7的输出通道 (512)
                                        # self.enc_channels_spec[-8] 是 enc_c1的输出通道 (32)
            
            skip_connection_ch = self.enc_channels_spec[skip_channel_idx] # 这是对应编码器层的输出通道
            upconv_out_ch = skip_connection_ch # 我们让上采样块的输出通道与跳跃连接的通道匹配
                                              # 这样拼接后卷积的输入通道数就是 2 * skip_connection_ch
            
            self.decoder_upconv_blocks.append(self._upconv_block(current_dec_channels, upconv_out_ch))
            
            # 拼接后的卷积层
            # 输入通道: upconv_out_ch + skip_connection_ch
            # 输出通道: 下一个上采样块的输入通道，或者是更小的值。
            # 如果是解码器的最后一个卷积块（对应编码器的第一个卷积块），输出通道数可以小一些（例如16或32）
            # 否则，输出通道数可以设置为下一个跳跃连接的通道数（即更浅一层编码器的输出通道）
            if i < num_decoder_stages - 1: # 不是最后一个拼接卷积块
                conv_after_skip_out_ch = self.enc_channels_spec[skip_channel_idx -1] # 目标是下一级跳跃连接的通道数
            else: # 是最后一个拼接卷积块 (对应enc_c1)
                conv_after_skip_out_ch = 32 # 例如，最后输出32通道，再用一个转置卷积到目标3通道
                                            # 或者直接在这里输出一个较小的通道数，如16
            
            self.decoder_conv_blocks.append(self._conv_block(upconv_out_ch + skip_connection_ch, conv_after_skip_out_ch, stride=1, padding=1))
            current_dec_channels = conv_after_skip_out_ch
            
        # 最后的转置卷积层，将特征图上采样到原始图像尺寸并调整通道数
        self.final_output_upconv = self._upconv_block(current_dec_channels, 16) # 例如，上采样到1024x1024，16通道
        self.final_output_conv = nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, no_bn_relu_for_last_conv=False):
        if no_bn_relu_for_last_conv and stride==2 : 
             return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        if stride == 1: 
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def _upconv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def encode(self, x):
        skip_outputs = []
        current_input = x
        for i, block in enumerate(self.encoder_blocks):
            current_input = block(current_input)
            if i < len(self.encoder_blocks) - 1: 
                skip_outputs.append(current_input)
        bottleneck_features = current_input
        flat = torch.flatten(bottleneck_features, start_dim=1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        self.skip_connections_for_decode = skip_outputs[::-1] 
        return mu, logvar

    def reparameterize(self, mu, logvar): std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std

    def decode(self, z):
        d_in = self.decoder_input_fc(z)
        num_channels_at_bottleneck = self.enc_channels_spec[-1]
        d_in = d_in.view(-1, num_channels_at_bottleneck, self.final_feature_map_size, self.final_feature_map_size)
        
        x = d_in
        for i in range(len(self.decoder_upconv_blocks)): # 应该是7个upconv+conv块
            x = self.decoder_upconv_blocks[i](x)
            skip_connection = self.skip_connections_for_decode[i]
            # 确保跳跃连接的空间维度与上采样后的x匹配
            if x.shape[2:] != skip_connection.shape[2:]:
                 x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder_conv_blocks[i](x)       
            
        x = self.final_output_upconv(x) # 最后一个上采样到全尺寸
        out = self.final_output_conv(x) 
        return self.sigmoid(out)

    def forward(self, x): mu,logvar = self.encode(x); z = self.reparameterize(mu,logvar); return self.decode(z), mu, logvar

vae_model_instance = UNetVAE(channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE[0]).to(train_main_device) 
vae_model_to_train = vae_model_instance 
if train_main_device.type == 'cuda' and torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU的DataParallel包装模型进行训练。")
    vae_model_to_train = nn.DataParallel(vae_model_instance, device_ids=train_device_ids_for_dp)
print(f"U-Net VAE模型已准备好在设备上训练。")

# -----------------------------------------------------------------------------
# 3. 损失函数和优化器 (使用BCE重构损失)
# -----------------------------------------------------------------------------
print("\n阶段 3: 定义损失函数和优化器 (使用BCE重构损失)")
def vae_loss_function(recon_x, x, mu, logvar, kld_weight_param=KLD_WEIGHT):
    calc_dev = x.device 
    recon_x, mu, logvar = recon_x.to(calc_dev), mu.to(calc_dev), logvar.to(calc_dev)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean') 
    kld_loss_summed = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    batch_size_effective = x.size(0) 
    kld_loss = kld_loss_summed / batch_size_effective 
    loss = recon_loss + kld_weight_param * kld_loss 
    return loss, recon_loss, kld_loss 
optimizer = optim.Adam(vae_model_to_train.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------------------------------
# 4. 训练循环
# -----------------------------------------------------------------------------
print("\n阶段 4: 开始训练U-Net VAE模型")
final_avg_train_loss = float('nan') 
if len(train_loader_normal) == 0: 
    print("严重错误: 训练数据加载器为空，无法开始训练。")
else:
    for epoch in range(EPOCHS): 
        vae_model_to_train.train(); train_loss_epoch, train_recon_loss_epoch, train_kld_loss_epoch = 0,0,0
        progress_bar = tqdm(train_loader_normal, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (data, _, _) in enumerate(progress_bar):
            if data is None: continue
            data = data.to(train_main_device) 
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model_to_train(data) 
            loss, recon_l, kld_l = vae_loss_function(recon_batch, data, mu, logvar, kld_weight_param=KLD_WEIGHT)
            if not torch.isfinite(loss): print(f"警告: Epoch {epoch+1}, Batch {batch_idx}, 非有限损失: {loss.item()}. 跳过。"); continue
            loss.backward(); optimizer.step()
            train_loss_epoch += loss.item() * len(data) 
            train_recon_loss_epoch += recon_l.item() * len(data)
            train_kld_loss_epoch += (kld_l.item() / (KLD_WEIGHT if KLD_WEIGHT > 1e-9 else 1.0)) * len(data) 
            progress_bar.set_postfix(loss=loss.item(), recon_loss_avg=recon_l.item(), kld_avg_unweighted=kld_l.item() / (KLD_WEIGHT if KLD_WEIGHT > 1e-9 else 1.0) )
        num_train_samples = len(train_dataset_normal) if train_dataset_normal and len(train_dataset_normal) > 0 else 1 
        final_avg_train_loss = train_loss_epoch/num_train_samples
        avg_recon_loss = train_recon_loss_epoch/num_train_samples
        avg_kld_loss_unweighted = train_kld_loss_epoch/num_train_samples 
        print(f"====> Epoch: {epoch+1} 平均损失: {final_avg_train_loss:.4f} (重构(BCE): {avg_recon_loss:.4f}, KL(unweighted_avg): {avg_kld_loss_unweighted:.4f})")
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1: 
            vae_model_to_train.eval() 
            with torch.no_grad():
                if len(train_loader_normal) > 0:
                    sample_data_iter = iter(train_loader_normal)
                    try:
                        sample_data,_,_ = next(sample_data_iter)
                        if sample_data is not None:
                            sample_data = sample_data.to(train_main_device) 
                            recon_sample,_,_ = vae_model_to_train(sample_data)
                            num_to_show = min(1, sample_data.size(0)) # BATCH_SIZE为1时，最多显示1张
                            vutils.save_image(torch.cat([sample_data[:num_to_show].cpu(),recon_sample[:num_to_show].cpu()]), 
                                              RESULTS_DIR/f"reconstruction_epoch_{epoch+1}_img1024_unet_vae.png", normalize=False)
                            print(f"重构样本保存至: {RESULTS_DIR/f'reconstruction_epoch_{epoch+1}_img1024_unet_vae.png'}")
                    except StopIteration: print("无法从训练加载器获取样本。")
                else: print("训练加载器为空，无法生成重构样本。")
    torch.save(vae_model_to_train.module.state_dict() if isinstance(vae_model_to_train, nn.DataParallel) else vae_model_to_train.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")

# -----------------------------------------------------------------------------
# 5. 缺陷检测与评估
# -----------------------------------------------------------------------------
print("\n阶段 5: 缺陷检测与评估")
if not MODEL_SAVE_PATH.exists():
    print(f"错误: 未找到模型 {MODEL_SAVE_PATH}。无法评估。")
else:
    print(f"从 {MODEL_SAVE_PATH} 加载模型用于评估...")
    eval_vae_model = UNetVAE(channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE[0]) 
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=eval_device) 
    eval_vae_model.load_state_dict(state_dict)
    eval_vae_model.to(eval_device); eval_vae_model.eval()
    print(f"评估模型已加载到 {eval_device} 并设为评估模式。")
    all_scores, all_labels, all_paths_viz, all_diff_maps_viz, all_recon_viz, all_orig_viz = [],[],[],[],[],[]
    if test_loader is None or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0 : 
        print("测试加载器为空或无数据，无法评估。")
    else:
        print("在测试集上计算异常分数...")
        with torch.no_grad():
            for images, labels, paths in tqdm(test_loader, desc="评估测试集"):
                if images is None: continue
                images = images.to(eval_device) 
                recon_images, mu, logvar = eval_vae_model(images) 
                for i in range(images.size(0)):
                    recon_error = F.binary_cross_entropy(recon_images[i], images[i], reduction='mean').item()
                    all_scores.append(recon_error); all_labels.append(labels[i].item())
                    all_paths_viz.append(paths[i]); orig_viz,recon_viz = images[i].cpu(),recon_images[i].cpu()
                    all_orig_viz.append(orig_viz); all_recon_viz.append(recon_viz)
                    all_diff_maps_viz.append(torch.abs(orig_viz-recon_viz).mean(dim=0).numpy()) 
        if not all_scores or not all_labels: print("未能计算测试分数，无法评估。")
        else:
            all_scores_np, all_labels_np = np.array(all_scores), np.array(all_labels)
            try: 
                roc_auc = roc_auc_score(all_labels_np, all_scores_np)
                print(f"\n测试集 ROC AUC: {roc_auc:.4f}")
                fpr, tpr, _ = roc_curve(all_labels_np, all_scores_np)
                plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC (AUC={roc_auc:.2f})')
                plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05])
                plt.xlabel('假阳性率'); plt.ylabel('真阳性率'); plt.title('ROC曲线'); plt.legend(loc="lower right"); plt.grid(True)
                plt.savefig(RESULTS_DIR/"roc_curve_img1024_unet_vae.png"); plt.show(); print(f"ROC图保存至: {RESULTS_DIR/'roc_curve_img1024_unet_vae.png'}")
                precision, recall, _ = precision_recall_curve(all_labels_np, all_scores_np)
                pr_auc = auc(recall, precision)
                print(f"测试集 PR AUC: {pr_auc:.4f}")
                plt.figure(figsize=(7,5)); plt.plot(recall,precision,color='blue',lw=2,label=f'PR (AUC={pr_auc:.2f})')
                plt.xlabel('召回率'); plt.ylabel('精确率'); plt.title('PR曲线'); plt.legend(loc="best"); plt.grid(True)
                plt.savefig(RESULTS_DIR/"pr_curve_img1024_unet_vae.png"); plt.show(); print(f"PR图保存至: {RESULTS_DIR/'pr_curve_img1024_unet_vae.png'}")
            except ValueError as e_roc: print(f"计算ROC/PR AUC时出错: {e_roc}. 可能测试集标签单一。")
            print("\n可视化测试样本及差异图...")
            viz_sample_indices = []; normal_indices_for_viz = [i for i,lab in enumerate(all_labels_np) if lab==0]; defect_indices_for_viz = [i for i,lab in enumerate(all_labels_np) if lab==1]
            if normal_indices_for_viz: viz_sample_indices.extend(random.sample(normal_indices_for_viz,min(3,len(normal_indices_for_viz))))
            if defect_indices_for_viz: viz_sample_indices.extend(random.sample(defect_indices_for_viz,min(3,len(defect_indices_for_viz))))
            if len(viz_sample_indices)<6 and len(all_paths_viz)>len(viz_sample_indices):
                remaining_indices = [i for i in range(len(all_paths_viz)) if i not in viz_sample_indices]
                if remaining_indices: viz_sample_indices.extend(random.sample(remaining_indices,min(6-len(viz_sample_indices),len(remaining_indices))))
            viz_sample_indices = sorted(list(set(viz_sample_indices))); num_viz = len(viz_sample_indices)
            if num_viz > 0:
                fig, axs = plt.subplots(num_viz,3,figsize=(12,num_viz*4)); axs = np.atleast_2d(axs) 
                for i_plot, data_idx in enumerate(viz_sample_indices):
                    current_path_str_viz = all_paths_viz[data_idx]; orig_plt = all_orig_viz[data_idx].permute(1,2,0).numpy(); recon_plt = all_recon_viz[data_idx].permute(1,2,0).numpy()
                    diff_map_to_show = all_diff_maps_viz[data_idx]; current_label_viz = all_labels_np[data_idx]; current_score_viz = all_scores_np[data_idx]
                    axs[i_plot,0].imshow(np.clip(orig_plt,0,1)); axs[i_plot,0].set_title(f"原图:{Path(current_path_str_viz).name}\n标签:{'缺陷'if current_label_viz==1 else'正常'}"); axs[i_plot,0].axis('off')
                    axs[i_plot,1].imshow(np.clip(recon_plt,0,1)); axs[i_plot,1].set_title(f"VAE重构"); axs[i_plot,1].axis('off')
                    im_d = axs[i_plot,2].imshow(diff_map_to_show,cmap='jet', vmin=0, vmax=np.percentile(diff_map_to_show,99) if diff_map_to_show.size>0 else 0.5); 
                    axs[i_plot,2].set_title(f"差异图(异常分:{current_score_viz:.4f})"); axs[i_plot,2].axis('off') 
                    fig.colorbar(im_d,ax=axs[i_plot,2],fraction=0.046,pad=0.04)
                plt.tight_layout(); plt.savefig(RESULTS_DIR/"test_samples_diff_img1024_unet_vae.png"); plt.show(); print(f"测试样本可视化保存至: {RESULTS_DIR/'test_samples_diff_img1024_unet_vae.png'}")
print("\n--- 脚本执行完毕 ---")