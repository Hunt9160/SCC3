import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import logging
from models.scc3 import SCC3


# -------------------------- 1. 配置与参数 --------------------------
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # 数据参数（单通道灰度图，适配满文图像）
        self.train_dir = cfg['data']['train_dir']
        self.val_dir = cfg['data']['val_dir']
        self.img_size = cfg['model']['img_size']  # [32, 128]
        self.batch_size = cfg['train']['batch_size']
        self.num_workers = cfg['train']['num_workers']

        # 模型参数（根据你的配置填写）
        self.in_channels = 1  # 满文图像为单通道灰度图
        self.out_char_num = cfg['model']['out_char_num']  # 32
        self.out_channels = cfg['model']['out_channels']  # 192
        self.nclass = cfg['model']['nclass']
        self.patch_merging = cfg['model']['patch_merging']  # 'Conv'
        self.embed_dim = cfg['model']['embed_dim']  # [96, 192, 256]
        self.depth = cfg['model']['depth']  # [3, 6, 6]
        self.num_heads = cfg['model']['num_heads']  # [3, 6, 8]
        self.mixer = cfg['model']['mixer']  # 列表中的'DPB'
        self.local_mixer = cfg['model']['local_mixer']  # [3,5,7]
        self.last_stage = cfg['model']['last_stage']  # True
        self.prenorm = cfg['model']['prenorm']  # True

        # 训练参数
        self.epochs = cfg['train']['epochs']
        self.lr = cfg['train']['lr']  # 0.0001
        self.beta1 = cfg['train']['beta1']
        self.beta2 = cfg['train']['beta2']
        self.weight_decay = cfg['train']['weight_decay']

        # 保存路径
        self.save_dir = cfg['path']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)


# -------------------------- 2. 数据集与数据加载 --------------------------
class SCC3Dataset(Dataset):
    def __init__(self, img_dir, img_size, transform=None):
        """文件名格式：标签_图像名.jpg"""
        self.img_dir = img_dir
        self.imgH, self.imgW = img_size
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

        # 字符集
        self.chars = "abcdefghijklmnopqrstuvwxyz'"
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 索引从1开始，0预留为空白符
        self.char2idx[''] = 0  # 空白符
        self.idx2char = {v: k for k, v in self.char2idx.items()}  # 新增索引到字符的映射

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # 单通道灰度图

        if self.transform:
            img = self.transform(img)

        # 标签转索引
        label_idx = [self.char2idx[c] for c in label if c in self.char2idx]
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        return img, label_idx


# -------------------------- 3. 训练逻辑 --------------------------
def calculate_accuracy(outputs, labels_tuple, idx2char):
    # outputs形状：[B, T, C] → 转为[T, B, C]用于解码
    outputs = outputs.transpose(0, 1)  # [T, B, C]
    _, preds = outputs.max(2)  # [T, B]
    preds = preds.transpose(1, 0).cpu().numpy()  # [B, T]

    correct = 0
    for i in range(len(labels_tuple)):
        # 解码预测结果（去重+去blank）
        pred_seq = []
        prev = -1
        for p in preds[i]:
            if p != prev and p != 0:
                pred_seq.append(p)
                prev = p
        # 解码真实标签
        true_seq = labels_tuple[i].cpu().numpy().tolist()
        if pred_seq == true_seq:
            correct += 1
    return correct / len(labels_tuple) if len(labels_tuple) > 0 else 0.0


def train():
    # 配置与日志
    cfg = Config(config_path='./configs/scc3.yaml')  # 模型配置文件
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger('SCC3-Train')

    # 设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
    ])

    # 数据集与加载器
    train_dataset = SCC3Dataset(
        img_dir=cfg.train_dir,
        img_size=cfg.img_size,
        transform=transform
    )
    val_dataset = SCC3Dataset(
        img_dir=cfg.val_dir,
        img_size=cfg.img_size,
        transform=transform
    )

    # 自定义collate_fn处理变长标签
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
        return imgs, labels, label_lens  # labels为元组

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    # 模型初始化 + 添加分类层
    class SCC3WithHead(nn.Module):
        def __init__(self, base_model, nclass):
            super().__init__()
            self.base_model = base_model
            # 将out_channels映射到字符类别数nclass
            self.classifier = nn.Linear(cfg.out_channels, nclass)

        def forward(self, x):
            features = self.base_model(x)  # [B, out_char_num, out_channels]
            logits = self.classifier(features)  # [B, out_char_num, nclass]
            return logits

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
        use_lenhead=False
    )
    model = SCC3WithHead(base_model, cfg.nclass).to(device)

    # 损失函数与优化器（AdamW+CTCLoss）
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # 学习率衰减

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.epochs}', ncols=100)
        for imgs, labels_tuple, label_lens in pbar:
            imgs = imgs.to(device)
            labels = torch.cat(labels_tuple).to(device)  # 拼接为一维张量
            label_lens = label_lens.to(device)
            B = imgs.size(0)

            optimizer.zero_grad()
            outputs = model(imgs)  # SCC3输出：[B, T, C]（T=out_char_num=32）

            # 适配CTC损失的输入格式：[T, B, C]
            log_probs = outputs.log_softmax(2).transpose(0, 1)  # [T, B, C]
            T = log_probs.size(0)  # T=32
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)

            # 计算损失
            loss = criterion(log_probs, labels, input_lens, label_lens)
            loss.backward()
            optimizer.step()

            # 累计损失与准确率
            train_loss += loss.item() * B
            batch_acc = calculate_accuracy(outputs.detach(), labels_tuple, train_dataset.idx2char)
            train_acc += batch_acc * B

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_acc': f'{batch_acc:.4f}'
            })

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_acc / len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels_tuple, label_lens in val_loader:
                imgs = imgs.to(device)
                labels = torch.cat(labels_tuple).to(device)
                label_lens = label_lens.to(device)
                B = imgs.size(0)

                outputs = model(imgs)  # [B, T, C]
                log_probs = outputs.log_softmax(2).transpose(0, 1)  # [T, B, C]
                T = log_probs.size(0)
                input_lens = torch.full((B,), T, dtype=torch.long, device=device)

                loss = criterion(log_probs, labels, input_lens, label_lens)
                val_loss += loss.item() * B

                # 计算验证准确率
                batch_acc = calculate_accuracy(outputs, labels_tuple, val_dataset.idx2char)
                val_acc += batch_acc * B

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_acc / len(val_dataset)

        # 日志与保存
        logger.info(
            f'Epoch {epoch + 1} | '
            f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | '
            f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}'
        )

        # 保存最佳模型（按验证准确率）
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'best_scc3.pth'))
            logger.info(f'Saved best model (Val Acc: {best_val_acc:.4f})')

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'latest_scc3.pth'))
        scheduler.step()

    logger.info(f'Training finished. Best Val Acc: {best_val_acc:.4f}')


if __name__ == '__main__':
    train()