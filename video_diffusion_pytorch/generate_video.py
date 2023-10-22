import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, video_tensor_to_gif

model = Unet3D(
    dim=64,
    # this must be set to True to auto-use the bert model dimensions
    use_bert_text_cond=True,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,    # height and width of frames
    num_frames=25,     # number of video frames
    timesteps=1000,   # number of steps
    loss_type='l1'    # L1 or L2
)

# video (batch, channels, frames, height, width)
videos = torch.randn(3, 3, 25, 32, 32)

text = [
    'car driving through a tunnel',
    'car driving on the road',
    'car driving through the forest'
]

loss = diffusion(videos, cond=text)
loss.backward()
# after a lot of training

sampled_videos = diffusion.sample(cond=text, cond_scale=2)
sampled_videos.shape  # (3, 3, 25, 32, 32)

for i, video in enumerate(sampled_videos):
    video_tensor_to_gif(video, f'sampled_video_{i}.gif')
