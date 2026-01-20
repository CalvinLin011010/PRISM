import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

# from model._modules import DynamicPatchAttention_Sinusoidal, DynamicPatchAttention_CPE   ## , MultiHeadAttention_SApatch1, DynamicPatchAttention

from model._loss import SpectralControlLoss1

import math

## 20251223
class PRISM(SequentialRecModel):
    def __init__(self, args):
        super(PRISM, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.apply(self.init_weights)



    def forward(self, input_ids, user_ids=None, all_sequence_output=False, epoch=None):
        extended_attention_mask = self.get_attention_mask(input_ids)
        padding_mask = input_ids == 0  ## [batch, seq_len]
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                padding_mask=padding_mask,
                                                epoch=epoch
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output, sequence_emb

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids, epoch):
        seq_output, target = self.forward(input_ids)  # 256 50 64

        seq_output_last = seq_output[:, -1, :]  # 256 64

        item_emb = self.item_embeddings.weight  # 12102 64
        logits = torch.matmul(seq_output_last, item_emb.transpose(0, 1))  # 256 12102   answers 256
        loss = nn.CrossEntropyLoss()(logits, answers)

        cluster_loss = 0.0
        n_layers = 0
        # loss_decouple = 0.0
        loss_align = 0.0
        if hasattr(self.item_encoder, 'blocks'):
            for block in self.item_encoder.blocks:
                # if hasattr(block.layer, 'loss_decouple'):
                #     cluster_loss += block.layer.loss_decouple
                #     loss_decouple += block.layer.loss_decouple
                if hasattr(block.layer, 'loss_align'):
                    cluster_loss += block.layer.loss_align
                    loss_align += block.layer.loss_align
                n_layers += 1
        
        if n_layers > 0:
            cluster_loss = cluster_loss / n_layers
            # Weight for clustering loss, defaulting to 0.1 if not specified
            # cluster_weight = getattr(self.args, 'cluster_loss_weight', 0.5)
            loss_all = self.args.loss_lambda * loss + (1 - self.args.loss_lambda) * cluster_loss  
            loss_dict = {"CEloss": loss.item(),
                            self.args.extend_loss_type: cluster_loss.item(), 
                            #  "loss_decouple": loss_decouple.item(),
                            "loss_align": loss_align.item()}


        return loss_all, loss_dict


class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, padding_mask=None, epoch=None):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask, padding_mask, epoch)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, padding_mask, epoch):
        layer_output = self.layer(hidden_states, attention_mask, padding_mask, epoch)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)

        # self.attention_sapatch_layer = DynamicPatchAttention_CPE(args)

        # Original MultiHeadAttention from bsarec.py
        self.attention_layer = MultiHeadAttention(args)

        self.alpha = args.alpha
        self.cluster_alpha = args.cluster_alpha

        self.use_sa = args.use_sa


        # ELCRec Clustering Module
        # Default num_intents to 16 if not present in args
        self.num_intents = getattr(args, 'num_intents', 16)
        self.elcm = ELCM(self.num_intents, args.hidden_size)

        self.loss_align = 0.0


    def forward(self, input_tensor, attention_mask, padding_mask, epoch=None):
        padding_len = padding_mask.sum(1)
        dsp = self.filter_layer(input_tensor, padding_len)
    

        # 1. Attention Features
        attn_out = self.attention_layer(input_tensor, attention_mask)
        
        # 2. Clustering Features - 自编码式投影重构
        # logits: [batch, seq, num_intents]
        # norm_centers: [num_intents, hidden]
        logits, norm_users, norm_centers = self.elcm(input_tensor)
    
        if padding_mask is not None:
            valid = (padding_mask == 0).unsqueeze(-1)
            norm_users = norm_users * valid.float()
            logits = logits * valid.float()


        # Reconstruction as clustering features - 被簇结构"过滤"后的特征; 可微分的软量化（Differentiable Soft Quantization）; 显式注入聚类归纳偏置（Explicit Clustering Inductive Bias）
        cluster_feat = torch.matmul(logits, norm_centers)
        # logits [batch, seq_len, num_intents]
        # norm_centers [num_intents, hidden]
        # cluster_feat [batch, seq_len, hidden]

        #### 上面两行代码的本质
        #      [batch, seq, hidden] @ [hidden, num_intents] -> [batch, seq, num_intents]
        # [batch, seq, num_intents] @ [num_intents, hidden] -> [batch, seq, hidden]
        


        # Compute and store loss
        loss_align = self.elcm.compute_clustering_loss(logits, norm_users, norm_centers, epoch)
        self.loss_align = loss_align

        # Fuse: Frequency + (Attention + Clustering)
        hidden_states = self.alpha / (1 + self.cluster_alpha) * dsp + \
                        (1 - self.alpha) / (1 + self.cluster_alpha) * attn_out + \
                        self.cluster_alpha / (1 + self.cluster_alpha) * cluster_feat




        return hidden_states
    




## 参考 src/model/Liu 等 - End-to-end Learnable Clustering for Intent Learning in Recommendation.pdf

class ELCM(nn.Module):
    def __init__(self, num_intents, hidden_dim, tau=0.5):
        super(ELCM, self).__init__()
        self.num_intents = num_intents
        self.hidden_dim = hidden_dim
        self.tau = tau
        
        # 核心创新点：将聚类中心初始化为可学习的神经网络参数
        # 这允许通过梯度下降进行"软微调"
        self.cluster_centers = nn.Parameter(
            torch.FloatTensor(num_intents, hidden_dim)
        )
        # 初始化策略至关重要，通常使用Xavier均匀分布 - 为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等。
        nn.init.xavier_uniform_(self.cluster_centers)

        ## 数值统计 用户序列特征（[b, 1, num_intent]） 和 其聚类特征的数值统计
        self.max_cluster_dict = {intent_idx: 0 for intent_idx in range(num_intents)}


    def forward(self, x):
        """
        前向传播：计算用户嵌入与聚类中心的交互
        x: [batch_size, seq_len, hidden_dim]
        """
        # 1. L2 归一化，投影到单位球面上
        # dim=-1 ensures normalization along the feature dimension for both 2D and 3D inputs
        norm_x = F.normalize(x, p=2, dim=-1)  ## norm_x -> torch.Size([256, 50, 64])
        norm_centers = F.normalize(self.cluster_centers, p=2, dim=-1)  ## 

        # 2. 计算相似度矩阵
        # [batch, seq, hidden] @ [hidden, num_intents] -> [batch, seq, num_intents]
        logits = torch.matmul(norm_x, norm_centers.T)

        return logits, norm_x, norm_centers


    def compute_clustering_loss(self, logits, norm_users, norm_centers, epoch):
        """
        计算批次化的用户序列和各个簇中心拉近的聚类损失
        """

        '''
        ##### 基于相似度的损失函数计算        
        # --- 对齐损失 (Alignment Loss) -
        # 找到每个token/user最接近的中心
        # logits: [batch, seq, num_intents]
        # max_sim: [batch, seq]
        max_sim, _ = torch.max(logits, dim=-1)

        # 最小化距离等价于最大化相似度
        # Loss = 1 - mean(max_sim)
        loss_align = 1.0 - torch.mean(max_sim)
        '''


        ### 参考 ELCRec中的对序列表征取均值，这里复用 - self.args.seq_representation_type == "mean":
        # 计算每个样本的序列表征 - 取均值
        seq_repr = torch.mean(norm_users, dim=1, keepdim=True)  ## seq_repr -> torch.Size([256, 1, 64])

        ### 数值统计 - 绘制聚类的分布
        with torch.no_grad():
            sim_cluster = torch.matmul(seq_repr, (norm_centers.T))  ## [batch, 1, num_intents]
            max_sim_idx = sim_cluster.squeeze(1).argmax(dim=-1)
            counts = torch.bincount(max_sim_idx, minlength=self.num_intents)
            for k in range(self.num_intents):
                self.max_cluster_dict[k] += int(counts[k].item())




        ## 计算用户序列表征和各个簇中心拉近的L2距离的聚类损失
        # seq_repr: [batch, 1, hidden]
        # norm_centers: [num_intents, hidden]
        # loss_align: [batch, num_intents]
        loss_align = torch.mean((seq_repr - norm_centers.unsqueeze(0)) ** 2, dim=-1)  ## loss_align -> torch.Size([batch, hidden])


        loss_align = torch.sum(torch.mean(loss_align, dim=-1))
        loss_align = loss_align / (self.num_intents * seq_repr.shape[0] * 1.0)


        return -1.0 * loss_align





class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

        ## 加一个可学习的层和weight来为每个样本取一个高低频的截断值

        self.freq_len = args.max_seq_length // 2 + 1
        
        # Dynamic frequency cutoff learning
        # 1. Learn frequency representation per sample from hidden dims
        self.freq_projector = nn.Linear(args.hidden_size, 1)
        
        # 2. Compress frequency profile to scalar cutoff per sample
        self.cutoff_generator = nn.Sequential(
            nn.Linear(self.freq_len, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, 1)
        )

        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))


        ### 数值统计
        self.c_hist = {i: 0 for i in range(args.max_seq_length // 2)}

        self.dynamic_c = None

        # self.x_fft_in = 0
        # self.x_fft_out = 0
        self.dynamic_mask = None  


    def forward(self, input_tensor, padding_len):
        """
        input_tensor: [batch, seq_len, hidden] - 未做过 Masking 的输入，在padding补零的位置上，hidden_size 还有残留 position_embeddings
        padding_len: [batch] - 每个样本左边补0的个数
        """        
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        ## 将padding置0的hidden_size 也置0 -- 20251222晚  -- 几乎没有提升的小trick，保留
        padding_mask = torch.arange(input_tensor.size(1), device=input_tensor.device).unsqueeze(0) >= padding_len.unsqueeze(1)
        input_tensor = input_tensor * padding_mask.unsqueeze(-1)


        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        '''
        ##### 没什么用
        ## 修正padding左侧补零带来的相位角偏移引发的实部虚部混合的问题，这里Bsarec没有拆分实部虚部，所以似乎影响不大
        # 1. 计算反向旋转因子 (Phase Correction Factor)
        # 目标: 消除 e^(-i * 2pi * k * d / N) 的影响
        # 对策: 乘以 e^(+i * 2pi * k * d / N)
        # 频率索引 k: [0, 1, ..., N//2]
        k = torch.arange(x.size(1), device=x.device).view(1, -1, 1) # [1, Freq, 1]
        # 补零长度 d: [B, 1, 1]
        d = padding_len.view(x.size(0), 1, 1)
        # 计算相位角 theta = 2 * pi * k * d / N
        theta = 2 * math.pi * k * d / input_tensor.size(1) 
        # 构造复数旋转因子 e^(i * theta) = cos(theta) + i * sin(theta)
        # 注意：这里是正的 theta，用于抵消原本的负移位
        rotation_factor = torch.complex(torch.cos(theta), torch.sin(theta))
        # 2. 执行相位校正
        # 利用广播机制：[B, F, D] * [B, F, 1]
        x = x * rotation_factor        
        '''

       # Calculate sample-specific dynamic cutoff
        # Use magnitude spectrum [B, F, H]
        x_mag = torch.abs(x) 
        
        # Learn representation: [B, F, H] -> [B, F, 1] -> [B, F]
        freq_repr = self.freq_projector(x_mag).squeeze(-1)
        
        # Generate cutoff logit: [B, F] -> [B, 1]
        cutoff_logit = self.cutoff_generator(freq_repr)  ## 单独样本，单独截断值
        
        # Constrain to valid range [1, freq_len-1] using sigmoid
        cutoff_prob = torch.sigmoid(cutoff_logit)
        # [ 下界   上界 ]
        # c = 1.0 + (self.freq_len - 2.0) * cutoff_prob
        c = 2.0 + ((self.freq_len // 2) - 2.0) * cutoff_prob


        ### 收集c的值用于统计分析
        ## 初始化时这里都是7.5左右
        c_int = torch.round(c).detach().to(torch.int64)
        c_int = torch.clamp(c_int, 1, self.freq_len)

        self.dynamic_c = c_int  ## 每个模块在每个batch中都会覆盖，


        vals = c_int.view(-1).tolist()
        for v in vals:
            self.c_hist[v] = self.c_hist.get(v, 0) + 1   



        c = c.view(batch, 1, 1) # [B, 1, 1]
        
        # Generate soft mask for differentiability
        # indices: [1, F, 1]
        freq_indices = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)  ## 索引 0到25
        
        # Soft mask: sigmoid(c - index)
        # Approaches 1 when index < c (low pass), 0 when index > c (high pass)  ## freq_indices 大于索引值，学习为low pass，小于索引值的，离索引值越远
        mask = torch.sigmoid(c - freq_indices)  ## 
        
        # Apply mask   -- self attention 是低通滤波器，抹平了高频
        low_pass = x * mask ## 过滤了低频，保留序列前面的特征 （高频）


        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
