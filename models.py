import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 散射中心层 (SC-layer)
# ==========================================
class SCLayer(nn.Module):
    """
    基于 ISTA (迭代收缩止阈算法) 展开的物理隐式层
    用于提取 HRRP 的主导散射中心位置与强度
    """
    def __init__(self, N=984, M=306, lambda_val=0.1, max_iter=50):
        super(SCLayer, self).__init__()
        self.N = N  # 距离单元数量
        self.M = M  # 频率分量数量
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        
        # 预定义傅里叶基矩阵 Φ 
        # (实际应用中应根据雷达具体的带宽和频率步进进行初始化)
        phi = torch.randn(M, N, dtype=torch.complex64)
        self.register_buffer('phi', phi)
        
        # 计算 Lipschitz 常数以设定步长
        self.step_size = 1.0 / (torch.linalg.norm(self.phi, ord=2)**2)

    def forward(self, r):
        # 输入 r: 接收到的复数频率响应 (Batch, M)
        batch_size = r.size(0)
        device = r.device
        
        # 初始化散射系数
        w = torch.zeros(batch_size, self.N, dtype=torch.complex64).to(device)
        phi_h = self.phi.t().conj()
        
        # 展开的 ISTA 迭代过程
        for _ in range(self.max_iter):
            # 1. 计算残差梯度
            error = torch.matmul(w, self.phi.t()) - r
            gradient = torch.matmul(error, phi_h.t())
            
            # 2. 梯度下降
            w_next = w - self.step_size * gradient
            
            # 3. 软阈值操作 (实现稀疏性约束)
            w_mag = torch.abs(w_next)
            new_mag = torch.relu(w_mag - self.lambda_val * self.step_size)
            w = torch.polar(new_mag, torch.angle(w_next))
            
        # 返回散射中心的幅度(强度)特征
        return torch.abs(w).float()


# ==========================================
# 2. 1-D ResNet 编码器网络 (Encoder Network)
# ==========================================
class ConvBlock(nn.Module):
    """一维卷积残差块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class EncoderNetwork(nn.Module):
    """将提取的物理特征映射为深度表示向量"""
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        # 对应论文中的 1x5, 1x7, 1x5 等卷积配置
        self.layer1 = ConvBlock(1, 64, kernel_size=5)
        self.layer2 = ConvBlock(64, 128, kernel_size=7, stride=2)
        self.layer3 = ConvBlock(128, 256, kernel_size=5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1024)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度 (Batch, 1, N)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        q = self.fc(x)
        # 将表示归一化至单位超球
        return F.normalize(q, p=2, dim=1)


# ==========================================
# 3. 目标方位监督对比损失 (TAS-Con Loss)
# ==========================================
class TargetAspectSupervisedContrastiveLoss(nn.Module):
    """
    修改后的监督对比损失
    旨在拉近同类但不同方位的样本，推开不同类别的样本
    """
    def __init__(self, temperature=0.07):
        super(TargetAspectSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        batch_size = z.shape(0)
        # 生成同类掩码矩阵
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # 计算点积相似度
        logits = torch.matmul(z, z.t()) / self.temperature
        exp_logits = torch.exp(logits)
        
        loss = 0
        valid_anchors = 0
        
        for i in range(batch_size):
            # 获取同类正样本索引（排除自身）
            pos_indices = (mask[i] == 1).nonzero(as_tuple=True)[0]
            pos_indices = pos_indices[pos_indices != i]
            # 获取负样本掩码
            neg_mask = (mask[i] == 0)
            
            if len(pos_indices) > 0:
                # 负样本的分母求和
                sum_neg = exp_logits[i][neg_mask].sum()
                # 计算论文中改进的损失公式
                row_loss = -torch.log(exp_logits[i][pos_indices] / (exp_logits[i][pos_indices] + sum_neg))
                loss += row_loss.mean()
                valid_anchors += 1
                
        if valid_anchors > 0:
            loss = loss / valid_anchors
        return loss


# ==========================================
# 4. 完整的 PriorK-NN 网络架构
# ==========================================
class PriorKNN(nn.Module):
    """集成 SC-layer、特征编码、分类与对比学习投影网络"""
    def __init__(self, num_classes):
        super(PriorKNN, self).__init__()
        self.sc_layer = SCLayer()
        self.encoder = EncoderNetwork()
        self.classifier = nn.Linear(1024, num_classes)
        
        # 投影网络 (仅在训练对比损失时使用)
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x, mode='train'):
        # 1. 提取稀疏散射中心特征
        h = self.sc_layer(x)
        
        # 2. 深度编码特征
        q = self.encoder(h)
        
        # 3. 分类预测
        logits = self.classifier(q)
        
        # 4. 训练模式下返回用于对比学习的投影向量 z
        if mode == 'train':
            z = F.normalize(self.projector(q), p=2, dim=1)
            return logits, z
            
        return logits