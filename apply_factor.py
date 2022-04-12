import argparse

import torch
from torchvision import utils

from model import Generator
from tqdm import trange
import cv2
import os


def random_style():
    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)
    direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    grid = utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"factor_out/{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )


def modify_style():
    for degree in trange(-int(args.degree), int(args.degree+1)):
        direction = degree * eigvec[:, args.index].unsqueeze(0)
        latent = torch.load(args.latent_file).unsqueeze(0)
        latent = g.get_latent(latent)
        item, _ = g(
            [latent + direction], truncation=args.truncation,
            truncation_latent=trunc, input_is_latent=True,
        ).squeeze(0)
        item = norm_ip(item, item.min(), item.max())
        item = item.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        cv2.imwrite(f"{args.save_img_path}/{count:06d}.png",
                    item[..., ::-1])
        count += 1


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(
        description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--latent_file",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt, map_location=torch.device(args.device))
    g = Generator(args.size, 512, 8,
                  channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)
    if args.latent_file is None:
        random_style()
    else:
        modify_style()
