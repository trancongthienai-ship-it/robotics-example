import torch
import torch.nn.functional as F
from pvht01 import VideoDiffusionActionModel


def compute_wam_flow_loss(
    model: VideoDiffusionActionModel,
    batch: dict,
    lambda_action: float = 1.0
) -> torch.Tensor:
    """
    Hỗ trợ hỗn hợp cả dữ liệu Video thuần (has_action = False) và dữ liệu Robot (has_action = True).
    """
    z_clean = batch["future_video_latents"] # [B, N_v, C_z]
    a_clean = batch["action_chunk"]         # [B, H_a, d_a]
    has_action = batch["has_action_mask"]   # [B] - Boolean tensor (1: dữ liệu robot, 0: video internet)
    B, device = z_clean.shape[0], z_clean.device

    # 1. Lấy mẫu nhiễu Gaussian
    eps_v = torch.randn_like(z_clean)
    eps_a = torch.randn_like(a_clean)

    # 2. Lấy mẫu bước thời gian độc lập (Split Timesteps) để dạy mô hình nhiều chế độ suy luận
    tau_v = torch.rand(B, device=device)
    tau_a = torch.rand(B, device=device)

    # Đối với các mẫu Video không có nhãn hành động, khóa tau_a = 0 (hoàn toàn là nhiễu Gaussian)
    tau_a = torch.where(has_action, tau_a, torch.zeros_like(tau_a))

    # 3. Nội suy tuyến tính theo Rectified Flow
    tau_v_exp = tau_v[:, None, None]
    tau_a_exp = tau_a[:, None, None]
    
    noisy_v = tau_v_exp * z_clean + (1.0 - tau_v_exp) * eps_v
    noisy_a = tau_a_exp * a_clean + (1.0 - tau_a_exp) * eps_a

    # 4. Forward pass
    pred_vel_v, pred_vel_a = model(
        noisy_v=noisy_v,
        noisy_a=noisy_a,
        tau_v=tau_v,
        tau_a=tau_a,
        obs_z0=batch["obs_z0"],
        proprio=batch["proprio"],
        lang_emb=batch["lang_emb"]
    )

    # 5. Tính hàm Loss vận tốc (Target Velocity = Clean - Noise)
    target_vel_v = z_clean - eps_v
    target_vel_a = a_clean - eps_a

    loss_video = F.mse_loss(pred_vel_v, target_vel_v)

    # Chỉ tính Action Loss trên các mẫu thực sự có nhãn điều khiển robot
    action_mse = F.mse_loss(pred_vel_a, target_vel_a, reduction="none").mean(dim=(1, 2)) # [B]
    if has_action.any():
        loss_action = (action_mse * has_action.float()).sum() / has_action.float().sum()
    else:
        loss_action = torch.tensor(0.0, device=device)

    return loss_video + lambda_action * loss_action