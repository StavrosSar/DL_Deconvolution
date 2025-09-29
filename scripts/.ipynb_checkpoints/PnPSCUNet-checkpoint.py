import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import argparse
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage import exposure

import deepinv as dinv
from deepinv.loss import R2RLoss
from deepinv.unfolded import unfolded_builder
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import GaussianNoise, Denoising
from deepinv.loss.metric import SSIM, PSNR

# =========================
# Configuration and Setup
# =========================
def setup_experiment():
    """Initialize experiment directory and parse arguments"""
    save_dir = "../results/PnPSCUNet_images_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    parser = argparse.ArgumentParser(description='Train PnP-SCUNet denoising model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--sigma', type=float, help='Override noise sigma value')
    parser.add_argument('--subset_size', type=int, help='Override dataset subset size')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.sigma is not None:
        config['physics']['sigma'] = args.sigma
    if args.subset_size is not None:
        config['data']['subset_size'] = args.subset_size
        
    return config, save_dir

# =========================
# Data Utilities
# =========================
class DenoisingDataset(Dataset):
    """Dataset for denoising task with noisy-clean image pairs"""
    def __init__(self, noisy_imgs, clean_imgs):
        self.noisy = noisy_imgs
        self.clean = clean_imgs

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

def prepare_tensor(data):
    """Convert numpy array to properly shaped torch tensor"""
    if data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        data = np.transpose(data, (0, 3, 1, 2))
    return torch.from_numpy(data).float()

def load_and_preprocess_data(config):
    """Load and preprocess training data"""
    x_train = np.load(config["data"]["x_train_path"])
    y_train = np.load(config["data"]["y_train_path"])
    
    subset_size = config["data"]["subset_size"]
    x_train = prepare_tensor(x_train)[:subset_size]
    y_train = prepare_tensor(y_train)[:subset_size]

    # Convert grayscale to 3-channel if needed
    if x_train.shape[1] == 1:
        x_train = x_train.repeat(1, 3, 1, 1)
        y_train = y_train.repeat(1, 3, 1, 1)

    # Normalize
    x_train = x_train / x_train.max()
    y_train = y_train / y_train.max()
    
    return x_train, y_train

# =========================
# Image Processing Utilities
# =========================
def adaptive_normalize(tensor):
    """Per-image normalization preserving relative intensities"""
    normalized = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        img = tensor[i]
        q_low = torch.quantile(img, 0.01)
        q_high = torch.quantile(img, 0.99)
        normalized[i] = (img - q_low) / (q_high - q_low + 1e-8)
    return normalized.clamp(0, 1)

def contrast_stretch(img):
    """Contrast stretching for single image"""
    img_np = img.cpu().numpy()
    p2, p98 = np.percentile(img_np, (2, 98))
    stretched = exposure.rescale_intensity(img_np, in_range=(p2, p98))
    return torch.from_numpy(stretched).to(img.device)

def prepare_for_display(img_tensor):
    """Convert tensor to properly scaled numpy array for display"""
    img = img_tensor.squeeze().cpu().numpy()
    if len(img.shape) == 3:
        img = np.mean(img, axis=0)
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

# =========================
# Visualization
# =========================
def display_results(images, titles, save_dir, colormap='inferno'):
    """Display and save comparison results"""
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        disp_img = prepare_for_display(img)
        im = ax.imshow(disp_img, cmap=colormap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    print(f"Saved comparison plot to: {comparison_path}")

def save_individual_images(images, titles, save_dir, colormap='inferno'):
    """Save individual result images"""
    for img, title in zip(images, titles):
        disp_img = prepare_for_display(img)
        filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        filepath = os.path.join(save_dir, filename)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(disp_img, cmap=colormap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(title)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
        print(f"Saved {title} to: {filepath}")

# =========================
# Evaluation Metrics
# =========================
def calculate_metrics(pred, target):
    """Calculate PSNR and L2 error metrics"""
    pred = pred.float().cpu()
    target = target.float().cpu()
    mse = torch.mean((pred - target) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    l2 = torch.norm(pred - target)
    return psnr.item(), l2.item()

def save_metrics(y, x, pred, final_img, training_time, save_dir):
    """Save evaluation metrics to text file"""
    psnr_raw, l2_raw = calculate_metrics(pred[0], x[0])
    psnr_enhanced, l2_enhanced = calculate_metrics(final_img, x[0])
    
    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("PnP-SCUNet Denoising Results\n")
        f.write("===========================\n\n")
        
        f.write("Value Ranges:\n")
        f.write(f"Noisy Input: {y[0].cpu().min().item():.3f} to {y[0].cpu().max().item():.3f}\n")
        f.write(f"Ground Truth: {x[0].cpu().min().item():.3f} to {x[0].cpu().max().item():.3f}\n")
        f.write(f"Raw Reconstruction: {pred[0].cpu().min().item():.3f} to {pred[0].cpu().max().item():.3f}\n")
        f.write(f"Enhanced Output: {final_img.cpu().min().item():.3f} to {final_img.cpu().max().item():.3f}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"Raw PSNR: {psnr_raw:.2f} dB\n")
        f.write(f"Enhanced PSNR: {psnr_enhanced:.2f} dB\n")
        f.write(f"L2 Error (Raw): {l2_raw:.3f}\n")
        f.write(f"L2 Error (Enhanced): {l2_enhanced:.3f}\n\n")
        
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        f.write(f"Total training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"Total training time in seconds: {training_time:.2f}s\n")

    print(f"Saved metrics to: {metrics_path}")
    return psnr_raw, psnr_enhanced, l2_raw, l2_enhanced

# =========================
# Model Evaluation
# =========================
def evaluate_model(model, physics, test_loader, device, training_time, save_dir):
    """Comprehensive model evaluation with visualization and metrics"""
    model.eval()
    with torch.no_grad():
        y, x = next(iter(test_loader))
        y, x = y.to(device), x.to(device)
        pred = model(y, physics)
        pred_normalized = adaptive_normalize(pred)
        final_img = contrast_stretch(pred_normalized[0].unsqueeze(0))[0]

    # Prepare results
    images = [y[0].cpu(), x[0].cpu(), pred[0].cpu(), final_img.cpu()]
    titles = ["Noisy Input", "Ground Truth", "Raw Reconstruction", "Enhanced Reconstruction"]

    # === NUMERICAL OUTPUTS ===
    # Flatten images for array printing
    flat_input = y[0].cpu().flatten().numpy()
    flat_target = x[0].cpu().flatten().numpy()
    flat_recon = pred[0].cpu().flatten().numpy()
    flat_enhanced = final_img.cpu().flatten().numpy()

    # Print array values
    np.set_printoptions(precision=6, threshold=20, linewidth=100)
    print("\nFlattened NumPy Arrays of Sample 0")
    print("\nInput (Noisy):")
    print(flat_input[:20])
    print("\nGround Truth:")
    print(flat_target[:20])
    print("\nRaw Reconstruction:")
    print(flat_recon[:20])
    print("\nEnhanced Reconstruction:")
    print(flat_enhanced[:20])

    # Print statistics
    print("\nValue Ranges:")
    print(f"Noisy Input: {y[0].cpu().min().item():.3f} to {y[0].cpu().max().item():.3f}")
    print(f"Ground Truth: {x[0].cpu().min().item():.3f} to {x[0].cpu().max().item():.3f}")
    print(f"Raw Reconstruction: {pred[0].cpu().min().item():.3f} to {pred[0].cpu().max().item():.3f}")
    print(f"Enhanced Output: {final_img.cpu().min().item():.3f} to {final_img.cpu().max().item():.3f}")

    # Calculate metrics
    psnr_raw, l2_raw = calculate_metrics(pred[0], x[0])
    psnr_enhanced, l2_enhanced = calculate_metrics(final_img, x[0])

    print(f"\nMetrics:")
    print(f"Raw PSNR: {psnr_raw:.2f} dB")
    print(f"Enhanced PSNR: {psnr_enhanced:.2f} dB")
    print(f"L2 Error (Raw): {l2_raw:.3f}")
    print(f"L2 Error (Enhanced): {l2_enhanced:.3f}")

    # Visualization
    display_results(images, titles, save_dir)
    save_individual_images(images, titles, save_dir)

    # Save metrics to file
    save_metrics(y, x, pred, final_img, training_time, save_dir)

    # Print summary
    print(f"\nMetrics Summary:")
    print(f"Raw PSNR: {psnr_raw:.2f} dB")
    print(f"Enhanced PSNR: {psnr_enhanced:.2f} dB")
    print(f"L2 Error (Raw): {l2_raw:.3f}")
    print(f"L2 Error (Enhanced): {l2_enhanced:.3f}")

# =========================
# Main Training Function
# =========================
def main():
    """Main training and evaluation pipeline"""
    config, save_dir = setup_experiment()
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Data loading
    x_train, y_train = load_and_preprocess_data(config)
    
    # Dataset preparation
    full_dataset = DenoisingDataset(y_train, x_train)
    train_size = int(config["train"]["train_split"] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False
    )

    # Model setup
    denoiser = dinv.models.SCUNet(pretrained=None, device=device)
    prior = PnP(denoiser=denoiser).to(device)
    physics = Denoising(
        noise_model=GaussianNoise(sigma=config["physics"]["sigma"])
    ).to(device)

    model = unfolded_builder(
        iteration="PGD",
        data_fidelity=L2(),
        max_iter=config["algo"]["max_iter"],
        prior=prior,
        trainable_params=config["algo"]["trainable_params"],
        params_algo=config["algo"]["params_algo"],
    ).to(device)

    model.enable_checkpointing = True

    # Optimization setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["train"]["eta_min"],
    )

    loss = R2RLoss()

    # Trainer setup
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=train_loader,
        epochs=config["train"]["epochs"],
        scheduler=scheduler,
        losses=loss,
        optimizer=optimizer,
        device=device,
        verbose=True,
        show_progress_bar=True,
        grad_clip=config["train"]["grad_clip"],
        wandb_vis=False,
        metrics=[SSIM(), PSNR()],
        plot_images=True,
        save_path=None,
        early_stop=True,
    )

    # Training
    print(f"\nStarting training with {len(train_dataset)} samples")
    start_time = time.time()
    model = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")

    # Evaluation
    trainer.test(test_loader, compare_no_learning=False)
    evaluate_model(model, physics, test_loader, device, training_time, save_dir)

if __name__ == "__main__":
    main()