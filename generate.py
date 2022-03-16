import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import os
import cv2
import numpy as np


def generate(args, g_ema, device, mean_latent):
    os.makedirs(args.save_img_path, exist_ok=True)

    g_ema.eval()
    with torch.no_grad():
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"{args.save_img_path}/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def generate_img(args, g_ema, device, mean_latent):
    os.makedirs(args.save_img_path, exist_ok=True)
    # output min max -1 ~ 1
    g_ema.eval()
    count = 1
    with torch.no_grad():
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            for item in sample:
                item = norm_ip(item, item.min(), item.max())
                item = item.mul(255).add_(0.5).clamp_(0, 255).permute(
                    1, 2, 0).to('cpu', torch.uint8).numpy()
                print(np.min(item), np.max(item))
                cv2.imwrite(f"{args.save_img_path}/{count:06d}.png",
                            item[..., ::-1])
                count += 1


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float,
                        default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--save_img_path",
        type=str,
        help="save  generateimage dir",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate_img(args, g_ema, device, mean_latent)
