import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class MMDiTBlock(nn.Module):
    """
    Khối Multi-Modal DiT xử lý đồng thời chuỗi Video Tokens và Action Tokens.
    Sử dụng cơ chế adaLN-Zero riêng biệt cho từng modality nhưng chia sẻ Joint Self-Attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm_v1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_a1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm_v2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_a2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.mlp_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Bộ sinh tham số điều chế (scale, shift, gate) riêng cho Video và Action
        self.adaLN_v = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.adaLN_a = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor, cond_v: torch.Tensor, cond_a: torch.Tensor):
        # 1. Trích xuất tham số điều chế từ timestep + context
        shift_msa_v, scale_msa_v, gate_msa_v, shift_mlp_v, scale_mlp_v, gate_mlp_v = (
            self.adaLN_v(cond_v).chunk(6, dim=-1)
        )
        shift_msa_a, scale_msa_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = (
            self.adaLN_a(cond_a).chunk(6, dim=-1)
        )

        # 2. Điều chế trước khi vào Joint Attention
        v_mod = self.norm_v1(x_v) * (1 + scale_msa_v.unsqueeze(1)) + shift_msa_v.unsqueeze(1)
        a_mod = self.norm_a1(x_a) * (1 + scale_msa_a.unsqueeze(1)) + shift_msa_a.unsqueeze(1)

        # 3. Joint Self-Attention giữa Video Tương lai và Chuỗi Hành động
        N_v = x_v.shape[1]
        joint_tokens = torch.cat([v_mod, a_mod], dim=1) # [B, N_v + H_a, D]
        attn_out, _ = self.attn(joint_tokens, joint_tokens, joint_tokens)
        
        attn_v, attn_a = attn_out[:, :N_v, :], attn_out[:, N_v:, :]

        # 4. Cập nhật nhánh Residual thứ nhất
        x_v = x_v + gate_msa_v.unsqueeze(1) * attn_v
        x_a = x_a + gate_msa_a.unsqueeze(1) * attn_a

        # 5. MLP riêng biệt cho từng modality
        v_mod2 = self.norm_v2(x_v) * (1 + scale_mlp_v.unsqueeze(1)) + shift_mlp_v.unsqueeze(1)
        a_mod2 = self.norm_a2(x_a) * (1 + scale_mlp_a.unsqueeze(1)) + shift_mlp_a.unsqueeze(1)

        x_v = x_v + gate_mlp_v.unsqueeze(1) * self.mlp_v(v_mod2)
        x_a = x_a + gate_mlp_a.unsqueeze(1) * self.mlp_a(a_mod2)

        return x_v, x_a


class VideoDiffusionActionModel(nn.Module):
    def __init__(
        self,
        video_latent_dim: int = 16,   # Số kênh của 3D VAE Latent
        num_video_tokens: int = 256,  # T_v * h * w sau khi patchify
        action_dim: int = 10,         # 3 pos + 6 rot (6D) + 1 gripper
        action_horizon: int = 16,     # Độ dài Action Chunk H_a
        proprio_dim: int = 10,        # Trạng thái khớp hiện tại q_0
        lang_dim: int = 768,          # Vector ngôn ngữ
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8
    ):
        super().__init__()
        self.num_video_tokens = num_video_tokens
        self.action_horizon = action_horizon

        # Nhúng đầu vào
        self.video_proj_in = nn.Linear(video_latent_dim, hidden_dim)
        self.action_proj_in = nn.Linear(action_dim, hidden_dim)
        
        # Positional Embeddings
        self.pos_emb_v = nn.Parameter(torch.randn(1, num_video_tokens, hidden_dim) * 0.02)
        self.pos_emb_a = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)

        # Bộ mã hóa thời gian khử nhiễu độc lập (Independent Timesteps tau_v, tau_a)
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_mlp_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_mlp_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        # Bộ mã hóa ngữ cảnh (Quan sát hiện tại z_0 + Ngôn ngữ + Proprioception)
        self.obs_cond_proj = nn.Linear(video_latent_dim, hidden_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(lang_dim + proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Mạng xương sống MM-DiT
        self.blocks = nn.ModuleList([MMDiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # Đầu ra dự đoán trường vận tốc (Velocity Field v_v, v_a cho Flow Matching)
        self.video_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, video_latent_dim))
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))

    def forward(
        self,
        noisy_v: torch.Tensor,   # [B, N_v, C_z] - Video tương lai đang nhiễu
        noisy_a: torch.Tensor,   # [B, H_a, d_a] - Hành động đang nhiễu
        tau_v: torch.Tensor,     # [B] - Timestep của video trong [0, 1]
        tau_a: torch.Tensor,     # [B] - Timestep của action trong [0, 1]
        obs_z0: torch.Tensor,    # [B, N_obs, C_z] - Khung hình hiện tại (sạch)
        proprio: torch.Tensor,   # [B, d_q]
        lang_emb: torch.Tensor   # [B, D_lang]
    ):
        # 1. Chiếu sang không gian ẩn và cộng Positional Embedding
        x_v = self.video_proj_in(noisy_v) + self.pos_emb_v
        x_a = self.action_proj_in(noisy_a) + self.pos_emb_a

        # Ghép thông tin ảnh hiện tại z_0 vào luồng video thông qua trung bình toàn cục hoặc nối chuỗi
        obs_summary = self.obs_cond_proj(obs_z0).mean(dim=1) # [B, D]
        global_ctx = self.context_proj(torch.cat([lang_emb, proprio], dim=-1)) + obs_summary

        # 2. Tạo vector điều chế độc lập cho Video (theo tau_v) và Action (theo tau_a)
        cond_v = self.time_mlp_v(self.time_emb(tau_v)) + global_ctx
        cond_a = self.time_mlp_a(self.time_emb(tau_a)) + global_ctx

        # 3. Đi qua chuỗi khối MM-DiT
        for block in self.blocks:
            x_v, x_a = block(x_v, x_a, cond_v, cond_a)

        # 4. Xuất trường vận tốc dự đoán (Velocity predictions)
        pred_vel_v = self.video_head(x_v)
        pred_vel_a = self.action_head(x_a)

        return pred_vel_v, pred_vel_a