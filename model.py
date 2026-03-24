import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from transformers import AutoModel


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        backbone      = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.cbam     = CBAM(channels=1792)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Sequential(
            nn.Linear(1792, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.features[-3:].parameters():
            p.requires_grad = True

    def forward(self, x):
        feat = self.features(x)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)


class CrossModalAttention(nn.Module):
    def __init__(self, img_dim=512, text_dim=768, num_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, text_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim   = text_dim,
            num_heads   = num_heads,
            dropout     = 0.1,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, img_emb, text_tokens, attention_mask=None):
        query        = self.img_proj(img_emb).unsqueeze(1)
        key_pad_mask = (attention_mask == 0) if attention_mask is not None else None
        attended, attn_weights = self.attn(
            query            = query,
            key              = text_tokens,
            value            = text_tokens,
            key_padding_mask = key_pad_mask,
        )
        out = self.norm(attended.squeeze(1))
        return out, attn_weights


class TextEncoder(nn.Module):
    def __init__(self, bert_model="dmis-lab/biobert-v1.1", embed_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_dim  = self.bert.config.hidden_size
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.bert.encoder.layer[-2:].parameters():
            p.requires_grad = True
        self.cross_attn = CrossModalAttention(embed_dim, bert_dim, 8)
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, input_ids, attention_mask, img_emb):
        bert_out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_feats = bert_out.last_hidden_state
        guided, attn_w = self.cross_attn(img_emb, token_feats, attention_mask)
        return self.proj(guided), attn_w


class DynamicGatedFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, img_emb, txt_emb):
        alpha = self.gate(torch.cat([img_emb, txt_emb], dim=-1))
        alpha = alpha * 0.6 + 0.2
        fused = alpha * img_emb + (1.0 - alpha) * txt_emb
        return self.proj(fused), alpha


class BreastCancerMultimodalNet(nn.Module):
    def __init__(self, num_classes=2, embed_dim=512, dropout=0.4,
                 bert_model="dmis-lab/biobert-v1.1"):
        super().__init__()
        self.image_enc  = ImageEncoder(embed_dim)
        self.text_enc   = TextEncoder(bert_model, embed_dim)
        self.fusion     = DynamicGatedFusion(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        img_emb               = self.image_enc(images)
        txt_emb, attn_weights = self.text_enc(input_ids, attention_mask, img_emb)
        fused, alpha          = self.fusion(img_emb, txt_emb)
        logits                = self.classifier(fused)
        return logits, alpha, attn_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = BreastCancerMultimodalNet(num_classes=2).to(device)
    images = torch.randn(2, 3, 224, 224).to(device)
    ids    = torch.randint(0, 28996, (2, 128)).to(device)
    mask   = torch.ones(2, 128, dtype=torch.long).to(device)
    logits, alpha, attn_w = model(images, ids, mask)
    print(f"Logits : {logits.shape}")
    print(f"Alpha  : {alpha.shape}")
    print("Model is correct!")
