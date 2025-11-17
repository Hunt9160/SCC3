import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import logging
from models.scc3 import SCC3  # 导入SCC3模型


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # 数据参数
        self.test_dir = cfg['data']['test_dir']  # 测试集路径
        self.img_size = cfg['model']['img_size']  # 图像尺寸 [H, W]
        self.batch_size = cfg['test']['batch_size']  # 批次大小
        self.num_workers = cfg['test']['num_workers']  # 数据加载线程

        # 模型参数
        self.in_channels = 1  # 输入通道数（灰度图）
        self.out_char_num = cfg['model']['out_char_num']  # 输出字符数
        self.out_channels = cfg['model']['out_channels']  # 特征通道数
        self.nclass = cfg['model']['nclass']  # 类别数（字符数+1）
        self.patch_merging = cfg['model']['patch_merging']
        self.embed_dim = cfg['model']['embed_dim']
        self.depth = cfg['model']['depth']
        self.num_heads = cfg['model']['num_heads']
        self.mixer = cfg['model']['mixer']
        self.local_mixer = cfg['model']['local_mixer']
        self.last_stage = cfg['model']['last_stage']
        self.prenorm = cfg['model']['prenorm']

        # 保存路径
        self.save_dir = cfg['path']['save_dir']  # 模型保存目录


class SCC3Dataset(Dataset):
    def __init__(self, img_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size  # [H, W]
        self.transform = transform

        self.samples = []
        for filename in os.listdir(img_dir):
            if not os.path.isfile(os.path.join(img_dir, filename)):
                continue
            try:
                label = filename.split('_')[0]  # 从文件名提取标签
                img_path = os.path.join(img_dir, filename)
                self.samples.append((img_path, label))
            except:
                print(f"Skipping invalid file: {filename}")

        # 字符映射表（与训练时保持一致）
        self.chars = "abcdefghijklmnopqrstuvwxyz'"
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.char2idx[''] = 0  # 空白符
        self.idx2char = {i: c for c, i in self.char2idx.items()}  # 索引到字符的映射

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # 单通道灰度图

        if self.transform:
            img = self.transform(img)

        return img, label, img_path  # 返回图像、原始标签和图像路径


# 定义带分类头的SCC3模型（与训练时保持一致）
class SCC3WithHead(nn.Module):
    def __init__(self, base_model, nclass):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.out_channels, nclass)

    def forward(self, x):
        features = self.base_model(x)  # [B, out_char_num, out_channels]
        logits = self.classifier(features)  # [B, out_char_num, nclass]
        return logits


def decode_prediction(preds, idx2char, blank=0):
    """将模型预测结果解码为文本（与CRNN解码逻辑一致）"""
    decoded_preds = []
    for seq in preds:
        # 移除连续重复和空白符
        seq = [s for s, _ in itertools.groupby(seq) if s != blank]
        text = ''.join([idx2char[i] for i in seq])
        decoded_preds.append(text)
    return decoded_preds


def test():
    # 配置与日志
    cfg = Config(config_path='./configs/scc3.yaml')  # 使用SCC3的配置文件
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger('SCC3-Test')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 数据预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),  # 使用配置中的图像尺寸
        transforms.ToTensor(),
    ])

    # 测试数据集与加载器
    test_dataset = SCC3Dataset(
        img_dir=cfg.test_dir,
        img_size=cfg.img_size,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 初始化基础模型
    base_model = SCC3(
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mixer=cfg.mixer,
        local_mixer=cfg.local_mixer,
        patch_merging=cfg.patch_merging,
        out_channels=cfg.out_channels,
        out_char_num=cfg.out_char_num,
        last_stage=cfg.last_stage,
        prenorm=cfg.prenorm,
        use_lenhead=False  # 与训练时保持一致
    )

    # 加载带分类头的模型
    model = SCC3WithHead(base_model, cfg.nclass).to(device)

    # 加载最佳模型权重
    model_path = os.path.join(cfg.save_dir, 'best_scc3.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f'Loaded model from {model_path}')
    else:
        raise FileNotFoundError(f'Model not found at {model_path}')

    model.eval()

    # 预测与评估
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for imgs, labels, img_paths in tqdm(test_loader, desc='Testing'):
            imgs = imgs.to(device)

            # 模型预测（SCC3输出格式：[B, out_char_num, nclass]）
            outputs = model(imgs)
            probs = outputs.softmax(2)  # 转换为概率分布

            # 获取预测结果（每个位置的最大概率类别）
            preds = probs.argmax(2)  # [B, out_char_num]
            preds = preds.cpu().numpy()  # 转为numpy数组

            # 解码预测结果
            decoded_preds = decode_prediction(preds, test_dataset.idx2char)

            # 计算准确率
            batch_size = len(labels)
            total_samples += batch_size
            correct_predictions += sum(1 for pred, gt in zip(decoded_preds, labels) if pred == gt)

            # 保存预测结果用于后续分析
            all_predictions.extend(decoded_preds)
            all_ground_truths.extend(labels)

    # 计算并输出准确率
    accuracy = correct_predictions / total_samples
    logger.info(f"识别准确率: {accuracy:.4f} ({correct_predictions}/{total_samples})")


if __name__ == '__main__':
    import itertools  # 用于解码时的groupby操作

    test()
