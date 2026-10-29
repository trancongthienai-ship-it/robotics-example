import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import subprocess
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import numpy as np
import cv2

from pvht01 import VideoDiffusionActionModel
from pvht02 import compute_wam_flow_loss


YOUTUBE_URL = "https://www.youtube.com/shorts/fkSximDCp50"


# =====================================================================
# 1. TẢI VIDEO YOUTUBE & CẮT ĐOẠN CHUYỂN ĐỘNG HỢP LÝ (KHÔNG CHUYỂN CẢNH)
# =====================================================================
def ensure_youtube_video_downloaded(url: str = YOUTUBE_URL) -> str:
    """
    Tìm file video YouTube đã tải trong thư mục hoặc tự động tải về bằng yt-dlp.
    """
    candidates = glob.glob("youtube_raw*.mp4")
    if candidates:
        return candidates[0]

    print(f"[1] Đang tải video từ YouTube: {url} ...")
    cmd = [sys.executable, "-m", "yt_dlp", "--js-runtimes", "node", "-o", "youtube_raw.mp4", url]
    subprocess.run(cmd, check=True)

    candidates = glob.glob("youtube_raw*.mp4")
    if not candidates:
        raise FileNotFoundError("Không tìm thấy file video mp4 sau khi tải bằng yt-dlp.")
    return candidates[0]


def cut_best_robot_segment(
    raw_video_path: str,
    clip_output_path: str = "youtube_clip.mp4",
    start_sec: float = 1.35,
    end_sec: float = 3.85,
) -> str:
    """
    Cắt đoạn video từ giây 1.35s -> 3.85s:
    Đây là cảnh góc máy cố định quay cánh tay robot hạ kẹp xuống gắp khối vuông
    màu xanh dương trên breadboard rồi nhấc lên (không bị cắt cảnh đột ngột).
    """
    cap = cv2.VideoCapture(raw_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_output_path, fourcc, fps, (width, height))

    frame_idx = 0
    saved_frames = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_idx <= end_frame:
            # Kiểm tra tránh lấy nhầm frame chuyển cảnh (jump cut) ở đầu/cuối đoạn
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
                if diff > 55.0 and saved_frames > 10:
                    # Dừng ngay nếu phát hiện chuyển cảnh sang góc máy khác
                    break
            prev_gray = gray
            writer.write(frame)
            saved_frames += 1

        frame_idx += 1
        if frame_idx > end_frame:
            break

    cap.release()
    writer.release()
    print(
        f"[1] Đã cắt đoạn video hợp lý ({start_sec:.2f}s -> {end_sec:.2f}s, {saved_frames} frames) "
        f"lưu tại: {os.path.abspath(clip_output_path)}"
    )
    return clip_output_path


# =====================================================================
# 2. PATCHIFY & UNPATCHIFY CHO KHUNG HÌNH CHỮ NHẬT (PORTRAIT 9:16)
# =====================================================================
def patchify_video(frames: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Chuyển chuỗi khung hình [B, T, C, H, W] (giá trị trong [-1, 1])
    thành chuỗi Video Latent Tokens [B, T * (H/p) * (W/p), C * p * p].
    """
    B, T, C, H, W = frames.shape
    p = patch_size
    h, w = H // p, W // p
    patches = frames.reshape(B, T, C, h, p, w, p).permute(0, 1, 3, 5, 2, 4, 6)
    return patches.reshape(B, T * h * w, C * p * p)


def unpatchify_video(
    tokens: torch.Tensor,
    num_frames: int,
    img_h: int = 144,
    img_w: int = 80,
    patch_size: int = 16,
    channels: int = 3,
) -> torch.Tensor:
    """
    Khôi phục chuỗi Video Latent Tokens [B, N_v, C_z]
    về lại chuỗi khung hình ảnh [B, T, C, H, W] (giá trị trong [-1, 1]).
    """
    B = tokens.shape[0]
    p = patch_size
    h, w = img_h // p, img_w // p
    patches = tokens.reshape(B, num_frames, h, w, channels, p, p)
    frames = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, num_frames, channels, img_h, img_w)
    return frames.clamp(-1.0, 1.0)


def load_youtube_clip_to_batch(
    video_path: str,
    device: str,
    num_future_frames: int = 6,
    img_h: int = 144,
    img_w: int = 80,
    patch_size: int = 16,
    action_horizon: int = 16,
    action_dim: int = 10,
) -> tuple[dict, torch.Tensor]:
    """
    Đọc file youtube_clip.mp4:
    - Lấy khung hình đầu tiên làm obs_z0 (ảnh hiện tại)
    - Lấy num_future_frames khung hình tiếp theo làm future_video_latents
    - Đặt has_action_mask = False (vì video YouTube không có nhãn góc khớp robot)
    """
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
        raw_frames.append(frame_resized)
    cap.release()

    if len(raw_frames) < num_future_frames + 1:
        raise ValueError(f"Video quá ngắn ({len(raw_frames)} frames).")

    # Lấy đều 1 khung hình đầu (obs_z0) + 6 khung hình tiếp theo (future_video_latents)
    indices = np.linspace(0, len(raw_frames) - 1, num_future_frames + 1, dtype=int)
    selected = np.stack([raw_frames[i] for i in indices], axis=0)  # [1 + T_v, H, W, 3]

    video_tensor = torch.from_numpy(selected).float() / 127.5 - 1.0
    video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1, 1 + T_v, 3, H, W]

    obs_frame = video_tensor[:, :1]      # [1, 1, 3, H, W]
    future_frames = video_tensor[:, 1:]  # [1, T_v, 3, H, W]

    obs_z0 = patchify_video(obs_frame, patch_size=patch_size)
    future_video_latents = patchify_video(future_frames, patch_size=patch_size)

    # Video YouTube không có nhãn điều khiển robot -> has_action_mask = False
    action_chunk = torch.zeros(1, action_horizon, action_dim, device=device)
    proprio = torch.zeros(1, 10, device=device)

    torch.manual_seed(42)
    lang_emb = torch.randn(1, 768, device=device)

    batch = {
        "future_video_latents": future_video_latents,
        "action_chunk": action_chunk,
        "has_action_mask": torch.tensor([False], device=device),  # False = Video Internet / YouTube
        "obs_z0": obs_z0,
        "proprio": proprio,
        "lang_emb": lang_emb,
    }
    return batch, video_tensor


# =====================================================================
# 3. SUY LUẬN (INFERENCE) & XUẤT VIDEO DỰ ĐOÁN
# =====================================================================
@torch.no_grad()
def sample_video_and_actions(
    model: VideoDiffusionActionModel,
    obs_z0: torch.Tensor,
    proprio: torch.Tensor,
    lang_emb: torch.Tensor,
    num_steps: int = 30,
    video_latent_dim: int = 768,
    action_dim: int = 10,
    has_action: bool = False,
):
    """
    Suy luận bằng phương pháp tích phân Euler ODE (tau: 0 -> 1)
    Từ khung hình đầu tiên obs_z0 của video YouTube, mô hình tự sinh ra các khung hình tiếp theo.
    """
    model.eval()
    B = obs_z0.shape[0]
    device = obs_z0.device

    z_v = torch.randn(B, model.num_video_tokens, video_latent_dim, device=device)
    z_a = torch.randn(B, model.action_horizon, action_dim, device=device)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = step / num_steps
        tau_v = torch.full((B,), t, device=device)
        # Với chế độ suy luận từ video thuần, khóa tau_a = 0.0 và giữ nguyên nhiễu z_a như lúc train
        tau_a = torch.full((B,), t if has_action else 0.0, device=device)

        pred_vel_v, pred_vel_a = model(
            noisy_v=z_v,
            noisy_a=z_a,
            tau_v=tau_v,
            tau_a=tau_a,
            obs_z0=obs_z0,
            proprio=proprio,
            lang_emb=lang_emb,
        )

        z_v = z_v + pred_vel_v * dt
        if has_action:
            z_a = z_a + pred_vel_a * dt

    return z_v, z_a


def save_results_to_mp4(
    gt_frames: torch.Tensor,
    pred_future_frames: torch.Tensor,
    pred_video_path: str = "predicted_output.mp4",
    compare_video_path: str = "comparison.mp4",
    out_w: int = 240,
    out_h: int = 432,
    fps: int = 4,
):
    """
    Xuất video dự đoán (predicted_output.mp4) và video so sánh song song (comparison.mp4).
    """
    full_pred = torch.cat([gt_frames[:, :1], pred_future_frames], dim=1)

    gt_np = ((gt_frames[0].permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pred_np = ((full_pred[0].permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_pred = cv2.VideoWriter(pred_video_path, fourcc, fps, (out_w, out_h))
    writer_cmp = cv2.VideoWriter(compare_video_path, fourcc, fps, (out_w * 2, out_h + 40))

    # Lặp lại 3 lần để video dài ~5 giây, dễ xem trên trình phát video
    for _ in range(3):
        for t in range(len(pred_np)):
            gt_bgr = cv2.cvtColor(gt_np[t], cv2.COLOR_RGB2BGR)
            pred_bgr = cv2.cvtColor(pred_np[t], cv2.COLOR_RGB2BGR)

            gt_up = cv2.resize(gt_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            pred_up = cv2.resize(pred_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            writer_pred.write(pred_up)

            canvas = np.zeros((out_h + 40, out_w * 2, 3), dtype=np.uint8)
            canvas[40:, :out_w] = gt_up
            canvas[40:, out_w:] = pred_up

            label_left = f"YouTube GT (Frame {t})"
            label_right = "Input z0" if t == 0 else f"Diffusion Pred ({t})"
            cv2.putText(canvas, label_left, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.putText(canvas, label_right, (out_w + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)
            writer_cmp.write(canvas)

    writer_pred.release()
    writer_cmp.release()
    print(f"[4] Đã xuất video mô hình dự đoán tại: {os.path.abspath(pred_video_path)}")
    print(f"[5] Đã xuất video đối chiếu tại      : {os.path.abspath(compare_video_path)}")


# =====================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_info = f"{device} ({torch.cuda.get_device_name(0)})" if device == "cuda" else device
    print(f"=== Huấn luyện Video Diffusion trên Video YouTube Shorts (Thiết bị: {device_info}) ===")

    # Bước 1: Dùng sẵn youtube_clip.mp4 nếu đã có trong repo, nếu chưa có mới tải từ YouTube và cắt
    clip_path = "youtube_clip.mp4"
    if os.path.exists(clip_path):
        print(f"[1] Đã tìm thấy sẵn file video mẫu: {os.path.abspath(clip_path)}")
    else:
        raw_video_path = ensure_youtube_video_downloaded(YOUTUBE_URL)
        clip_path = cut_best_robot_segment(raw_video_path, clip_path, start_sec=1.35, end_sec=3.85)

    # Bước 2: Đọc đoạn clip đã cắt và chuyển thành các Latent Patch Tokens
    # Tự động tăng độ phân giải & số bước train khi phát hiện GPU (RTX 3060)
    img_h, img_w = (144, 80) if device == "cuda" else (72, 40)
    patch_size = 8
    num_future_frames = 4
    video_latent_dim = 3 * patch_size * patch_size  # 192
    num_video_tokens = num_future_frames * (img_h // patch_size) * (img_w // patch_size)

    single_batch, gt_video_tensor = load_youtube_clip_to_batch(
        video_path=clip_path,
        device=device,
        num_future_frames=num_future_frames,
        img_h=img_h,
        img_w=img_w,
        patch_size=patch_size,
    )
    batch_size = 32 if device == "cuda" else 16
    batch = {
        k: v.repeat(batch_size, *([1] * (v.ndim - 1)))
        for k, v in single_batch.items()
    }
    print(
        f"[2] Đã mã hóa '{clip_path}' ({img_w}x{img_h}) -> obs_z0: {tuple(single_batch['obs_z0'].shape)}, "
        f"future_video_latents: {tuple(single_batch['future_video_latents'].shape)}, "
        f"has_action_mask: {single_batch['has_action_mask'].tolist()}"
    )

    # Bước 3: Khởi tạo mô hình và huấn luyện với compute_wam_flow_loss
    model = VideoDiffusionActionModel(
        video_latent_dim=video_latent_dim,
        num_video_tokens=num_video_tokens,
        action_dim=10,
        action_horizon=16,
        proprio_dim=10,
        lang_dim=768,
        hidden_dim=384 if device == "cuda" else 256,
        depth=6 if device == "cuda" else 4,
        num_heads=8,
    ).to(device)

    num_train_steps = 1000 if device == "cuda" else 300
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)

    print(f"\n[3] Bắt đầu huấn luyện ({num_train_steps} steps, batch_size={batch_size}) trên đoạn video YouTube...")
    model.train()
    log_interval = 100 if device == "cuda" else 50
    for step in range(1, num_train_steps + 1):
        optimizer.zero_grad()
        loss = compute_wam_flow_loss(model, batch, lambda_action=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % log_interval == 0 or step == num_train_steps:
            print(f"    Step {step:4d}/{num_train_steps} | Video Flow Loss: {loss.item():.4f}")

    # Bước 4: Suy luận (Inference) - Chỉ đưa vào Frame 0, mô hình tự dự đoán các khung hình tiếp theo
    print("\n--- Đang chạy Inference (Euler ODE Solver từ nhiễu Gaussian) ---")
    pred_video_tokens, _ = sample_video_and_actions(
        model=model,
        obs_z0=single_batch["obs_z0"],
        proprio=single_batch["proprio"],
        lang_emb=single_batch["lang_emb"],
        num_steps=50,
        video_latent_dim=video_latent_dim,
        has_action=False,
    )

    pred_future_frames = unpatchify_video(
        pred_video_tokens,
        num_frames=num_future_frames,
        img_h=img_h,
        img_w=img_w,
        patch_size=patch_size,
    )

    save_results_to_mp4(gt_video_tensor, pred_future_frames)
    print("\nHoàn tất! Bạn có thể mở 'youtube_clip.mp4' và 'comparison.mp4' để xem kết quả.")


if __name__ == "__main__":
    main()
