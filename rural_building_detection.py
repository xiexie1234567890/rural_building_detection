import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, f1_score
import datetime

# 设置随机种子以保证实验可重复性
torch.manual_seed(42)
np.random.seed(42)

# -------------------------- 1. 数据加载模块 --------------------------
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像和掩码
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转为灰度图
        
        # 预处理转换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # 二值化掩码
        mask = (mask > 0.5).float()
        
        return image, mask

# -------------------------- 2. 数据预处理模块 --------------------------
def get_transforms(image_size=256):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 掩码的转换（不需要归一化）
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return train_transform, val_test_transform, mask_transform

# -------------------------- 3. 模型构建模块 --------------------------

# 3.3 多尺度混合注意力模块 (Multi-scale Hybrid Attention Module)
class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out) * x

class MultiScaleHybridAttention(nn.Module):
    """多尺度混合注意力模块"""
    def __init__(self, in_channels):
        super(MultiScaleHybridAttention, self).__init__()
        
        # 并行多尺度卷积
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        
        # 通道注意力
        self.channel_att = ChannelAttention(in_channels)
        
        # 空间注意力
        self.spatial_att = SpatialAttention()
        
        # 融合卷积
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 并行多尺度特征提取
        feat3x3 = self.relu(self.conv3x3(x))
        feat5x5 = self.relu(self.conv5x5(x))
        feat7x7 = self.relu(self.conv7x7(x))
        
        # 通道注意力处理
        feat3x3 = self.channel_att(feat3x3)
        feat5x5 = self.channel_att(feat5x5)
        feat7x7 = self.channel_att(feat7x7)
        
        # 特征融合
        fused = torch.cat([feat3x3, feat5x5, feat7x7], dim=1)
        fused = self.relu(self.fusion_conv(fused))
        
        # 空间注意力处理
        out = self.spatial_att(fused)
        
        return out + x  # 残差连接

# 3.4 动态特征金字塔网络 (Dynamic Feature Pyramid Network)
class DynamicFeaturePyramid(nn.Module):
    """动态特征金字塔网络"""
    def __init__(self, in_channels_list, out_channels):
        super(DynamicFeaturePyramid, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 动态权重生成器
        self.dynamic_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid()
            ) for in_ch in in_channels_list
        ])
        
        # 特征融合卷积
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) 
            for in_ch in in_channels_list
        ])
        
        # 上采样层
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True)
            for i in range(len(in_channels_list))
        ])
    
    def forward(self, features):
        assert len(features) == len(self.in_channels_list)
        
        # 计算动态权重
        weights = [dw(f) for dw, f in zip(self.dynamic_weights, features)]
        
        # 归一化权重
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # 应用权重并融合特征
        fused_features = []
        for f, w, conv, upsample in zip(features, weights, self.fusion_convs, self.upsamples):
            weighted_feat = f * w
            conv_feat = conv(weighted_feat)
            upsampled_feat = upsample(conv_feat)
            fused_features.append(upsampled_feat)
        
        # 求和融合
        out = sum(fused_features)
        
        return out

# 3.5 渐进对比学习 (Progressive Contrastive Learning) 策略
class ProgressiveContrastiveLoss(nn.Module):
    """渐进对比学习损失"""
    def __init__(self, temperature=0.1):
        super(ProgressiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, features, labels, epoch, max_epochs):
        # 计算特征之间的相似度
        features = torch.nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 生成标签掩码
        labels = labels.view(-1)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # 对角线元素设为0（排除自身比较）
        mask = mask.fill_diagonal_(0)
        
        # 渐进难度调整
        progress = epoch / max_epochs
        # 随着训练进行，逐渐增加正样本权重
        positive_weight = 1.0 + progress * 2.0
        
        # 损失计算
        logits = similarity_matrix
        targets = mask * positive_weight
        
        # 对称损失
        loss = self.cross_entropy(logits, targets)
        
        return loss

# 主模型类
class RuralBuildingDetector(nn.Module):
    """农村建筑检测模型"""
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', num_classes=1):
        super(RuralBuildingDetector, self).__init__()
        
        # 编码器（不使用预训练权重，加快测试速度）
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=None  # 不使用预训练权重
        )
        
        # 多尺度混合注意力模块
        self.msha = MultiScaleHybridAttention(self.encoder.out_channels[-1])
        
        # 动态特征金字塔网络
        self.dfpn = DynamicFeaturePyramid(self.encoder.out_channels, 256)
        
        # 最终分类头
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()  # 二分类输出
        )
    
    def forward(self, x):
        # 编码器特征提取
        encoder_features = self.encoder(x)
        
        # 多尺度混合注意力处理
        msha_features = [self.msha(feat) for feat in encoder_features]
        
        # 动态特征金字塔融合
        dfpn_out = self.dfpn(msha_features)
        
        # 解码器输出
        out = self.decoder(dfpn_out)
        
        return out

# -------------------------- 4. 模型训练模块 --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    model.to(device)
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算指标
            train_loss += loss.item() * images.size(0)
            
            outputs_np = outputs.detach().cpu().numpy() > 0.5
            masks_np = masks.detach().cpu().numpy()
            
            train_precision += precision_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
            train_recall += recall_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
            train_f1 += f1_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
        
        # 训练损失和指标平均
        train_loss /= len(train_loader.dataset)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_f1 /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, masks)
                
                # 计算指标
                val_loss += loss.item() * images.size(0)
                
                outputs_np = outputs.detach().cpu().numpy() > 0.5
                masks_np = masks.detach().cpu().numpy()
                
                val_precision += precision_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
                val_recall += recall_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
                val_f1 += f1_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
        
        # 验证损失和指标平均
        val_loss /= len(val_loader.dataset)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_rural_building_model.pth')
        
        # 记录历史
        train_history.append({'loss': train_loss, 'precision': train_precision, 'recall': train_recall, 'f1': train_f1})
        val_history.append({'loss': val_loss, 'precision': val_precision, 'recall': val_recall, 'f1': val_f1})
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
    
    return train_history, val_history

# -------------------------- 5. 模型验证模块 --------------------------
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_f1 = 0.0
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 计算指标
            test_loss += loss.item() * images.size(0)
            
            outputs_np = outputs.detach().cpu().numpy() > 0.5
            masks_np = masks.detach().cpu().numpy()
            
            test_precision += precision_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
            test_recall += recall_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
            test_f1 += f1_score(masks_np.flatten(), outputs_np.flatten(), zero_division=0)
    
    # 平均指标
    test_loss /= len(test_loader.dataset)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_f1 /= len(test_loader)
    
    return {
        'loss': test_loss,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }

# -------------------------- 6. 结果输出模块 --------------------------
def save_results_to_txt(results, filename='model_results.txt'):
    with open(filename, 'w') as f:
        f.write("农村建筑图像识别模型结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n测试集指标:\n")
        f.write(f"损失: {results['loss']:.4f}\n")
        f.write(f"精确率 (Precision): {results['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {results['recall']:.4f}\n")
        f.write(f"F1分数: {results['f1']:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("模型架构: 基于多尺度混合注意力和动态特征金字塔网络的农村建筑检测模型\n")
        f.write("数据集: Massachusetts Buildings Dataset\n")
        f.write("任务: 二分类建筑分割\n")

# -------------------------- 主函数 --------------------------
def main():
    # 数据集路径
    data_root = r'data\Massachusetts Buildings Dataset'
    train_image_dir = os.path.join(data_root, 'Training Set', 'Input images')
    train_mask_dir = os.path.join(data_root, 'Training Set', 'Target maps')
    val_image_dir = os.path.join(data_root, 'Validation Set', 'Input images')
    val_mask_dir = os.path.join(data_root, 'Validation Set', 'Target maps')
    test_image_dir = os.path.join(data_root, 'Test Set', 'Input images')
    test_mask_dir = os.path.join(data_root, 'Test Set', 'Target maps')
    
    # 参数设置
    image_size = 256
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据转换
    train_transform, val_test_transform, mask_transform = get_transforms(image_size)
    
    # 创建数据集
    train_dataset = BuildingDataset(train_image_dir, train_mask_dir, transform=train_transform)
    val_dataset = BuildingDataset(val_image_dir, val_mask_dir, transform=val_test_transform)
    test_dataset = BuildingDataset(test_image_dir, test_mask_dir, transform=val_test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")
    
    # 初始化模型
    model = RuralBuildingDetector()
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练模型
    print("开始训练模型...")
    train_history, val_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_rural_building_model.pth'))
    
    # 模型验证
    print("\n开始验证模型...")
    test_results = evaluate_model(model, test_loader, device)
    
    # 输出结果
    print("\n测试结果:")
    print(f"损失: {test_results['loss']:.4f}")
    print(f"精确率: {test_results['precision']:.4f}")
    print(f"召回率: {test_results['recall']:.4f}")
    print(f"F1分数: {test_results['f1']:.4f}")
    
    # 保存结果到txt文件
    save_results_to_txt(test_results)
    print("\n结果已保存到 model_results.txt 文件中")

if __name__ == '__main__':
    main()