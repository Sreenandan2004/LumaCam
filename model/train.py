"""
LumaCam GAN Training & Validation Pipeline
==========================================
Main training orchestrator for the Exposure-Conditioned Attention U-Net GAN:
- Alternating Discriminator (PatchGAN) and Generator (Attention U-Net) optimization
- Mixed-Precision Training (AMP) with torch.cuda.amp.GradScaler for 4GB VRAM constraint
- Composite Objective: L1 Reconstruction + MS-SSIM + Conditional Adversarial Loss
- PSNR & SSIM validation benchmarking and best-checkpoint persistence
- Dry-run mode support using synthetic Poisson-Gaussian sensor data

Usage:
    # Full dataset training
    python model/train.py --config model/configs/train_config.yaml

    # Fast local dry-run (e.g., verifying GTX 1650 pipeline)
    python model/train.py --config model/configs/train_config.yaml --dry-run --epochs 2 --batch-size 1
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.networks.generator import ExposureConditionedAttentionUNet
from model.networks.discriminator import PatchGANDiscriminator
from model.losses.composite_loss import CompositeLoss, AdversarialLoss, MSSSIMLoss
from model.datasets.sid_dataset import SIDSonyDataset, DummySIDDataset


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula:
        PSNR = 20 * log10(MAX / sqrt(MSE))
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse))).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LumaCam Attention U-Net GAN Training Loop")
    parser.add_argument("--config", type=str, default="model/configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic DummySIDDataset for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cuda' or 'cpu')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    config: Dict[str, Any], dry_run: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Prepares training and validation DataLoaders."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 0)
    patch_size = data_cfg.get("patch_size", 512)

    if dry_run:
        print("[LumaCam] Initializing synthetic DummySIDDataset for dry-run verification...")
        train_dataset = DummySIDDataset(num_samples=16, patch_size=patch_size, seed=42)
        val_dataset = DummySIDDataset(num_samples=4, patch_size=patch_size, seed=1337)
    else:
        data_root = data_cfg.get("data_root", "data/Sony")
        train_list = data_cfg.get("train_list", None)
        val_list = data_cfg.get("val_list", None)

        if not Path(data_root).exists():
            print(
                f"[LumaCam WARNING] Data root '{data_root}' not found. Falling back to DummySIDDataset."
            )
            train_dataset = DummySIDDataset(num_samples=16, patch_size=patch_size, seed=42)
            val_dataset = DummySIDDataset(num_samples=4, patch_size=patch_size, seed=1337)
        else:
            train_dataset = SIDSonyDataset(
                data_root=data_root,
                list_file=train_list,
                patch_size=patch_size,
                is_train=True,
            )
            val_dataset = SIDSonyDataset(
                data_root=data_root,
                list_file=val_list,
                patch_size=patch_size,
                is_train=False,
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(len(train_dataset) > batch_size),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def validate(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates validation set returning mean PSNR and mean SSIM."""
    generator.eval()
    psnr_scores = []
    ssim_scores = []
    ssim_metric = MSSSIMLoss().to(device)

    with torch.no_grad():
        for bayer_in, gt_rgb, alpha in val_loader:
            bayer_in = bayer_in.to(device)
            gt_rgb = gt_rgb.to(device)
            alpha = alpha.to(device)

            pred_rgb, _ = generator(bayer_in, alpha)
            pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)

            psnr = calculate_psnr(pred_rgb, gt_rgb)
            psnr_scores.append(psnr)

            # Compute SSIM: 1 - MSSSIMLoss
            try:
                ms_loss = ssim_metric(pred_rgb, gt_rgb).item()
                ssim = max(0.0, 1.0 - ms_loss)
                ssim_scores.append(ssim)
            except Exception:
                ssim_scores.append(0.0)

    mean_psnr = float(sum(psnr_scores) / max(len(psnr_scores), 1))
    mean_ssim = float(sum(ssim_scores) / max(len(ssim_scores), 1))
    generator.train()
    return mean_psnr, mean_ssim


def train_lumacam():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"============================================================")
    print(f" LumaCam: Training Exposure-Conditioned Attention U-Net GAN")
    print(f" Device: {device} | Dry-run: {args.dry_run}")
    if device.type == "cuda":
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"============================================================")

    # Instantiate Models
    model_cfg = config.get("model", {})
    generator = ExposureConditionedAttentionUNet(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 3),
        base_channels=model_cfg.get("base_channels", 32),
        cond_dim=model_cfg.get("cond_dim", 1),
        use_pixel_shuffle=model_cfg.get("use_pixel_shuffle", True),
    ).to(device)

    # Discriminator receives 7 channels: 4 (Bayer raw upsampled) + 3 (sRGB)
    discriminator = PatchGANDiscriminator(in_channels=7, base_channels=64).to(device)

    # Loss Functions
    train_cfg = config.get("training", {})
    composite_loss = CompositeLoss(
        lambda_l1=train_cfg.get("lambda_l1", 100.0),
        lambda_ms_ssim=train_cfg.get("lambda_ms_ssim", 10.0),
        lambda_adv=train_cfg.get("lambda_adv", 1.0),
    ).to(device)
    adv_loss = AdversarialLoss().to(device)

    # Optimizers
    lr_g = float(train_cfg.get("lr_generator", 1e-4))
    lr_d = float(train_cfg.get("lr_discriminator", 1e-4))
    beta1 = float(train_cfg.get("beta1", 0.5))
    beta2 = float(train_cfg.get("beta2", 0.999))

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    # Automatic Mixed Precision (AMP)
    use_amp = bool(train_cfg.get("use_amp", True)) and (device.type == "cuda")
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(config, dry_run=args.dry_run)

    # Checkpoint Directory
    ckpt_cfg = config.get("checkpointing", {})
    save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = ckpt_cfg.get("save_every_n_epochs", 10)

    best_psnr = -1.0
    start_epoch = 1
    total_epochs = train_cfg.get("epochs", 200)

    # Resume if requested
    if args.resume and Path(args.resume).exists():
        print(f"[LumaCam] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_psnr = checkpoint.get("best_psnr", -1.0)

    log_cfg = config.get("logging", {})
    log_every = log_cfg.get("log_every_n_steps", 10)
    val_every = log_cfg.get("val_every_n_epochs", 1)

    print(f"\n[LumaCam] Starting training: {total_epochs} epochs, {len(train_loader)} batches/epoch...\n")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        generator.train()
        discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_l1 = 0.0
        running_ms = 0.0

        for step, (bayer_in, gt_rgb, alpha) in enumerate(train_loader, start=1):
            bayer_in = bayer_in.to(device)
            gt_rgb = gt_rgb.to(device)
            alpha = alpha.to(device)

            # -------------------------------------------------------------
            # Step 1: Update Discriminator (PatchGAN)
            # -------------------------------------------------------------
            optimizer_d.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Real pair classification
                real_pred = discriminator(bayer_in, gt_rgb)

                # Fake pair classification (detach fake image to avoid backprop into G)
                with torch.no_grad():
                    fake_rgb, _ = generator(bayer_in, alpha)

                fake_pred = discriminator(bayer_in, fake_rgb.detach())
                d_loss = adv_loss.discriminator_loss(real_pred, fake_pred)

            scaler_d.scale(d_loss).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # -------------------------------------------------------------
            # Step 2: Update Generator (Exposure Attention U-Net)
            # -------------------------------------------------------------
            optimizer_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                gen_rgb, _ = generator(bayer_in, alpha)
                # Discriminator judgment on freshly generated fake
                disc_fake_pred = discriminator(bayer_in, gen_rgb)

                # Composite loss: L1 + MS-SSIM + GAN
                g_losses = composite_loss(gen_rgb, gt_rgb, disc_fake_pred)
                g_loss = g_losses["total"]

            scaler_g.scale(g_loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            running_l1 += g_losses["l1"].item()
            running_ms += g_losses["ms_ssim"].item()

            if step % log_every == 0 or step == len(train_loader):
                print(
                    f"Epoch [{epoch:03d}/{total_epochs:03d}] | Step [{step:03d}/{len(train_loader):03d}] | "
                    f"D_Loss: {d_loss.item():.4f} | G_Total: {g_loss.item():.4f} | "
                    f"L1: {g_losses['l1'].item():.4f} | MS-SSIM: {g_losses['ms_ssim'].item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_d = running_d_loss / len(train_loader)
        avg_g = running_g_loss / len(train_loader)
        print(f"--> Epoch {epoch} complete in {epoch_time:.2f}s | Avg D_Loss: {avg_d:.4f} | Avg G_Loss: {avg_g:.4f}")

        # -------------------------------------------------------------
        # Validation & Checkpointing
        # -------------------------------------------------------------
        if epoch % val_every == 0 or epoch == total_epochs:
            val_psnr, val_ssim = validate(generator, val_loader, device)
            print(f"[Validation Epoch {epoch}] PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

            # Checkpoint metadata
            ckpt_state = {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_psnr": max(best_psnr, val_psnr),
                "config": config,
            }

            # Save latest
            torch.save(ckpt_state, save_dir / "lumacam_latest.pt")

            # Save best if PSNR improves
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(ckpt_state, save_dir / "lumacam_best.pt")
                print(f"[BEST] New best checkpoint saved with PSNR: {best_psnr:.2f} dB!")

        if epoch % save_every == 0:
            torch.save(ckpt_state, save_dir / f"lumacam_epoch_{epoch:03d}.pt")

    print(f"\n[LumaCam] Training finished successfully! Best Validation PSNR: {best_psnr:.2f} dB\n")


if __name__ == "__main__":
    train_lumacam()
