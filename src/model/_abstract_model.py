import torch
import torch.nn as nn
from model._modules import LayerNorm
from torch.nn.init import xavier_uniform_

class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.batch_size = args.batch_size

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()  ## [256, 50]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  #  [256, 1, 1, 50]    torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)  ## (1, 50, 50)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 ## 只保留上三角全1，主对角线也是0 （diagonal=1：表示从主对角线（index 0）向右上方偏移 1 个单位）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  ## [1, 1, 50, 50]  上三角（不包含主对角线）全False
        subsequent_mask = subsequent_mask.long().to(item_seq.device)  ## [1, 1, 50, 50]

        extended_attention_mask = extended_attention_mask * subsequent_mask  ## [256, 1, 1, 50]序列mask * [1, 1, 50, 50] 上三角（不包含主对角线）全False
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  ## 结果形状与数值符合常见 Transformer 的约定：允许位置为 0，不允许位置为一个很大的负数（-10000），用于在 softmax 前加到注意力分数上。
        ## [256, 1, 50, 50]  -- 生成一步异步的attention mask
        return extended_attention_mask 

    def forward(self, input_ids, all_sequence_output=False):
        pass

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)

    def calculate_loss(self, input_ids, answers):
        pass

