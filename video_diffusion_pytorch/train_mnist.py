import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pdb


class MyMovingMNIST(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64))])

        self.dataset = datasets.moving_mnist.MovingMNIST(
            root=root, download=True, transform=self.transform)

    def __getitem__(self, index):
        data = self.dataset[index]  # Ignore the target labels
        # Rearrange the dimensions from "TCHW" to "CFHW"
        data = data.permute(1, 0, 2, 3)
        return data

    def __len__(self):
        return len(self.dataset)


def video_tensor_to_gif(
        tensor, path, duration=120, loop=0, optimize=True):
    images = map(transforms.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(
        path, save_all=True, append_images=rest_imgs, duration=duration,
        loop=loop, optimize=optimize)
    return images


print("Loading Unet3D...")
model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

print("Loading GaussianDiffusion...")
diffusion = GaussianDiffusion(
    model,
    channels=1,
    image_size=64,
    num_frames=20,
    timesteps=1000,   # number of steps
    loss_type='l1'    # L1 or L2
).cuda()

print("Loading Trainer...")
transform = transforms.Compose([
    transforms.Resize((64, 64))])

trainer = Trainer(
    diffusion,
    './data',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    ds=MyMovingMNIST('./data'),
    train_batch_size=1,
    train_lr=1e-4,
    save_and_sample_every=1000,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True                        # turn on mixed precision
)

print("Training...")
trainer.train()

print("Sampling...")
sampled_videos = diffusion.sample(batch_size=4)
u_sampled_videos = sampled_videos.unbind(dim=1)
for i in range(len(u_sampled_videos)):
    images = video_tensor_to_gif(
        u_sampled_videos[i],
        "result_" + str(i) + ".gif")
