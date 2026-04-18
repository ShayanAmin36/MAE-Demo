# app.py — fixed: single MAE definition matching notebook's saved weights
import json, torch, gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# ── positional embedding (must come before MAE) ───────────────
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    half = embed_dim // 2
    g = np.arange(grid_size, dtype=np.float32)
    grid_h, grid_w = np.meshgrid(g, g, indexing="ij")
    grid = np.stack([grid_h.ravel(), grid_w.ravel()], axis=0)
    omega = np.arange(half // 2, dtype=np.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)
    out_h = np.einsum("n,d->nd", grid[0], omega)
    out_w = np.einsum("n,d->nd", grid[1], omega)
    emb = np.concatenate([
        np.sin(out_h), np.cos(out_h),
        np.sin(out_w), np.cos(out_w),
    ], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)


# ── transformer sub-modules ───────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Must match notebook exactly: fc1 → act → fc2 (named fc1/fc2, not 'net')."""
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden    = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── MAE (single definition, matches notebook) ─────────────────

class MAE(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3, mask_ratio=0.75,
        enc_dim=768,  enc_depth=12,  enc_heads=12,
        dec_dim=384,  dec_depth=12,  dec_heads=6,
        mlp_ratio=4.0, **_,          # **_ absorbs any extra config keys safely
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size  = patch_size
        self.mask_ratio  = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim   = in_chans * patch_size * patch_size
        self.grid_size   = img_size // patch_size

        # encoder
        self.patch_embed = nn.Linear(self.patch_dim, enc_dim, bias=True)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.register_buffer("enc_pos_embed",
                             get_2d_sincos_pos_embed(enc_dim, self.grid_size))
        self.enc_blocks  = nn.ModuleList([
            TransformerBlock(enc_dim, enc_heads, mlp_ratio) for _ in range(enc_depth)
        ])
        self.enc_norm    = nn.LayerNorm(enc_dim)

        # projection
        self.enc_to_dec  = nn.Linear(enc_dim, dec_dim, bias=True)

        # decoder
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos_embed",
                             get_2d_sincos_pos_embed(dec_dim, self.grid_size))
        self.dec_blocks  = nn.ModuleList([
            TransformerBlock(dec_dim, dec_heads, mlp_ratio) for _ in range(dec_depth)
        ])
        self.dec_norm    = nn.LayerNorm(dec_dim)
        self.dec_head    = nn.Linear(dec_dim, self.patch_dim, bias=True)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, h * w, C * p * p)

    def unpatchify(self, patches):
        p, g, C = self.patch_size, self.grid_size, 3
        B = patches.shape[0]
        x = patches.reshape(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, g * p, g * p)

    def encode(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        x = self.patch_embed(self.patchify(imgs)) + self.enc_pos_embed
        B, N, D = x.shape
        n_vis = int(N * (1 - mask_ratio))

        noise        = torch.rand(B, N, device=x.device)
        ids_shuf     = noise.argsort(1)
        ids_restore  = ids_shuf.argsort(1)
        ids_vis      = ids_shuf[:, :n_vis]

        x_vis = x.gather(1, ids_vis.unsqueeze(-1).expand(-1, -1, D))
        mask  = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_vis, 0)

        cls   = self.cls_token.expand(B, -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)
        for blk in self.enc_blocks:
            x_vis = blk(x_vis)
        return self.enc_norm(x_vis), mask, ids_restore

    def decode(self, x_enc, ids_restore):
        x     = self.enc_to_dec(x_enc)
        x_tok = x[:, 1:]                              # drop CLS
        B, n_vis, D = x_tok.shape
        N      = ids_restore.shape[1]
        n_mask = N - n_vis

        mt     = self.mask_token.expand(B, n_mask, -1)
        x_full = torch.cat([x_tok, mt], 1)
        x_full = x_full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.dec_pos_embed
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        return self.dec_head(self.dec_norm(x_full))

    @torch.no_grad()
    def reconstruct(self, imgs, mask_ratio=0.75):
        x_enc, mask, ids_restore = self.encode(imgs, mask_ratio)
        pred = self.decode(x_enc, ids_restore)
        return pred, mask


# ── load model ────────────────────────────────────────────────
DEVICE = torch.device("cpu")

with open("mae_config.json") as f:
    cfg = json.load(f)

model = MAE(**cfg)
state = torch.load("mae_final.pth", map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()
print("Model loaded ✓")

# ── preprocessing helpers ─────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)

def make_masked(imgs, mask, ps=16):
    g   = 224 // ps
    out = imgs.clone()
    for idx in range(mask.shape[1]):
        if mask[0, idx] == 1:
            r, c = idx // g, idx % g
            out[0, :, r*ps:(r+1)*ps, c*ps:(c+1)*ps] = 0.0
    return out


# ── inference ─────────────────────────────────────────────────
def infer(pil_img, mask_ratio):
    if pil_img is None:
        return None, None, None
    # ensure RGB (handles RGBA or grayscale uploads)
    pil_img  = pil_img.convert("RGB")
    img_t    = preprocess(pil_img).unsqueeze(0)       # (1,3,224,224)

    pred, mask = model.reconstruct(img_t, float(mask_ratio))
    recon      = model.unpatchify(pred)
    masked     = make_masked(img_t, mask)

    return (
        Image.fromarray(denorm(masked[0])),
        Image.fromarray(denorm(recon[0])),
        Image.fromarray(denorm(img_t[0])),
    )


# ── Gradio UI ─────────────────────────────────────────────────
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0.1, maximum=0.95, value=0.75, step=0.05,
                  label="Masking Ratio"),
    ],
    outputs=[
        gr.Image(label="Masked Input"),
        gr.Image(label="MAE Reconstruction"),
        gr.Image(label="Original"),
    ],
    title="Masked Autoencoder (MAE) Demo",
    description=(
        "Upload any image. The model randomly masks the selected fraction of "
        "16×16 patches and reconstructs them using a Vision Transformer."
    ),
    flagging_mode="never",   
)

demo.launch()