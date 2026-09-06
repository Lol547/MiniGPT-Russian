import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0  # будет переопределено при создании модели


class PositionalEmbedding(nn.Module):    
    def __init__(self, sequence_length, vocab_size, output_dim, dropout=0.1, pad_id=PAD_ID):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, output_dim, padding_idx=pad_id)
        self.position_embeddings = nn.Embedding(sequence_length, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        embeddings = self.token_embeddings(inputs) + self.position_embeddings(positions)
        return self.dropout(embeddings)


class TransformerDecoderLayer(nn.Module):    
    def __init__(self, hidden_dim, intermediate_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.self_attention_layernorm = nn.LayerNorm(hidden_dim)
        
        self.feed_forward_1 = nn.Linear(hidden_dim, intermediate_dim)
        self.feed_forward_2 = nn.Linear(intermediate_dim, hidden_dim)
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        norm_x = self.self_attention_layernorm(x)
        attn_out, _ = self.self_attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        norm_x = self.feed_forward_layernorm(x)
        ffn_out = self.feed_forward_1(norm_x)
        ffn_out = F.gelu(ffn_out)
        ffn_out = self.feed_forward_2(ffn_out)
        ffn_out = self.dropout(ffn_out)
        x = x + ffn_out
        return x


class MiniGPT(nn.Module):    
    def __init__(
        self,
        vocab_size,
        sequence_length,
        pad_id=0,
        hidden_dim=512,
        intermediate_dim=2048,
        num_heads=8,
        num_layers=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding = PositionalEmbedding(
            sequence_length, vocab_size, hidden_dim, dropout, pad_id
        )
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, intermediate_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layernorm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.classifier.weight = self.embedding.token_embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        pad_id = self.embedding.token_embeddings.padding_idx
        x = self.embedding(inputs)
        padding_mask = (inputs == pad_id)
        
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        x = self.final_layernorm(x)
        logits = self.classifier(x)
        return logits
