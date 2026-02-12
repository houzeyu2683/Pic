import torch
import numpy
import torchvision
import sys
sys.path.insert(0, '.depcate')
import sparrow

# 載入 VAE decoder
device = 'cpu'
vae = sparrow.Model(device)
vae.activateLayer()
vae.loadVersion('sparrow-v1.0.0')
vae.eval()

# 載入 latent vectors (同一個影片的 frames)
compression = torch.load(
    './material/storage/lipread_pt/ABOUT/test/ABOUT_00001.pt',
    weights_only=True
)
assert isinstance(compression, torch.Tensor)

# 選兩個 frame 的 latent vector 做插值
index_a = 0
index_b = compression.size(0) - 1
z_a = compression[index_a]  # (C, H, W)
z_b = compression[index_b]  # (C, H, W)

# 線性插值產生中間 frames
steps = 30
frames = []
for i in range(steps):
    alpha = i / (steps - 1)
    z_t = (1 - alpha) * z_a + alpha * z_b
    with torch.no_grad():
        reconstruction = vae.getReconstruction(z_t[None, :, :, :])
    frame = reconstruction.squeeze(0).clamp(0, 1)
    frames += [frame]
    continue

# 存成圖片 grid 預覽
grid = torchvision.utils.make_grid(frames, nrow=10, padding=2)
torchvision.utils.save_image(grid, 'interpolation_grid.png')
print(f'已儲存 interpolation_grid.png ({steps} frames)')

# 存成影片
video = torch.stack(frames)  # (T, C, H, W)
video = (video * 255).to(torch.uint8)
video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
torchvision.io.write_video(
    'interpolation.mp4',
    video,
    fps=15
)
print('已儲存 interpolation.mp4')
