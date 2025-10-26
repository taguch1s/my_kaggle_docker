import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class JigsawModelTextW(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_layers: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states": True, "num_labels": num_labels})
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.regressor = nn.Linear(self.config.hidden_size, num_labels)

        # Cross attention: Body attends to Rule
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Feed-Forward層（Transformerスタイル）
        self.cross_attn_norm = nn.LayerNorm(self.config.hidden_size)
        self.cross_attn_dropout = nn.Dropout(0.1)
        # 🔥 4つの特徴量を結合するためのFeed-Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 4),  # 拡張
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
        )
        self.ff_norm = nn.LayerNorm(self.config.hidden_size)
        self.ff_dropout = nn.Dropout(0.1)

        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_layers:
            self._freeze_top_half_layers()

    def _freeze_top_half_layers(self):
        """先頭半分の層をfreeze"""
        n_layers = self.config.num_hidden_layers
        freeze_count = n_layers // 2 + 1

        print(f"Freezing top {freeze_count} layers out of {n_layers} total layers")

        for i in range(freeze_count):
            for name, param in self.backbone.encoder.layer[i].named_parameters():
                param.requires_grad = False

    def _extract_text_tokens(self, token_type_ids, attention_mask, last_hidden_state):
        """token_type_idsを使用してtext部分を抽出"""
        batch_size = token_type_ids.size(0)
        text_embeddings = []

        for i in range(batch_size):
            # text部分（token_type_id == 1）のマスクを作成
            text_mask = (token_type_ids[i] == 1) & (attention_mask[i] == 1)

            if text_mask.any():
                # text部分の埋め込みを抽出
                text_tokens = last_hidden_state[i][text_mask]
                text_pooled = text_tokens.mean(dim=0)
            else:
                text_pooled = torch.zeros(self.config.hidden_size, device=token_type_ids.device)

            text_embeddings.append(text_pooled)

        return torch.stack(text_embeddings)


    def _pool_by_token_type(self, token_type_ids, attention_mask, last_hidden_state):
        # last_hidden_state: [B,S,H], attention_mask: [B,S], token_type_ids: [B,S] or None
        use_type_ids = (
            token_type_ids is not None
            and token_type_ids.dim() == 2
            and torch.any(token_type_ids > 0)
        )
        if use_type_ids:
            text_mask_2d = (token_type_ids == 1) & (attention_mask == 1)  # [B,S] boolean AND
            rule_mask_2d = (token_type_ids == 0) & (attention_mask == 1)
        else:
            raise Exception

        # text
        text_mask = text_mask_2d.unsqueeze(-1).type_as(last_hidden_state)      # [B,S,1] -> broadcast OK
        text_num = (last_hidden_state * text_mask).sum(dim=1)                  # [B,H]
        text_feat = text_num / text_mask.sum(dim=1).clamp(min=1e-9)                        # [B,1]

        # rule
        rule_mask = rule_mask_2d.unsqueeze(-1).type_as(last_hidden_state)      # [B,S,1] -> broadcast OK
        rule_num = (last_hidden_state * rule_mask).sum(dim=1)                  # [B,H]
        rule_feat = rule_num / rule_mask.sum(dim=1).clamp(min=1e-9)

        return rule_feat, text_feat

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  # DeBERTa/RoBERTaはNoneでOK
        labels=None
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,  # 念のため明示
            return_dict=True
        )
        total_features, _ = outputs.last_hidden_state.max(1)  # tuple: [layer0..last]

        # Text部分のみのpooling
        rule_features, text_features = self._pool_by_token_type(
            token_type_ids, attention_mask, outputs.last_hidden_state
        )  # [batch_size, hidden_size]

        # Option 1: 現在の方式（pooled features同士のattention）
        attended_text, _ = self.cross_attention(
            query=text_features.unsqueeze(1),
            key=rule_features.unsqueeze(1),
            value=text_features.unsqueeze(1),
        )
        attended_text = attended_text.squeeze(1)

        # concat
        # Transformer-style processing
        attended_text = self.cross_attn_norm(attended_text)
        attended_text = self.cross_attn_dropout(attended_text)

        ff_output = self.feed_forward(torch.cat([attended_text, total_features], dim=1))
        final_features = self.ff_norm(ff_output)
        final_features = self.ff_dropout(final_features)

        logits = self.regressor(final_features)

        loss = None
        if labels is not None:
            # 分類（0/1）のときはCrossEntropy
            loss = self.loss_fn(logits, labels.long())

        return SequenceClassifierOutput(loss=loss, logits=logits)
