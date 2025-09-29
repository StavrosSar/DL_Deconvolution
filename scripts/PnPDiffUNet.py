import yaml
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from deepinv.physics import Denoising, GaussianNoise
from deepinv.loss.metric import SSIM, PSNR
from deepinv.loss import R2RLoss
from skimage import exposure

def setup_experiment():
    """Initialize experiment directory and parse arguments"""
    save_dir = "../results/PnPDiffUNet_images_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    parser = argparse.ArgumentParser(description='Train PnPDiffUNet denoising model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
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
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.sigma is not None:
        config['physics']['sigma'] = args.sigma
    if args.subset_size is not None:
        config['data']['subset_size'] = args.subset_size
        
    return config, save_dir, args.mode

# Dataset
class DenoisingDataset(Dataset):
    def __init__(self, noisy_imgs, clean_imgs):
        self.noisy = noisy_imgs
        self.clean = clean_imgs

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

def prepare_tensor(data):
    if data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        data = np.transpose(data, (0, 3, 1, 2))
    return torch.from_numpy(data).float()

# Enhancement & Visualization
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

def calculate_metrics(pred, target):
    """Calculate PSNR, SSIM and L2 error metrics"""
    pred = pred.float().cpu().unsqueeze(0)  # add batch for SSIM
    target = target.float().cpu().unsqueeze(0)

    psnr_metric = PSNR()
    ssim_metric = SSIM()

    psnr = psnr_metric(pred, target).item()
    ssim = ssim_metric(pred, target).item()
    l2 = torch.norm(pred - target).item()

    return psnr, ssim, l2

def save_metrics(y, x, pred, final_img, training_time, save_dir):
    """Save evaluation metrics to text file"""
    psnr_raw, ssim_raw, l2_raw = calculate_metrics(pred[0], x[0])
    psnr_enhanced, ssim_enhanced, l2_enhanced = calculate_metrics(final_img, x[0])

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("DiffUNet Denoising Results\n")
        f.write("===========================\n\n")
        
        f.write("Value Ranges:\n")
        f.write(f"Noisy Input: {y[0].cpu().min().item():.3f} to {y[0].cpu().max().item():.3f}\n")
        f.write(f"Ground Truth: {x[0].cpu().min().item():.3f} to {x[0].cpu().max().item():.3f}\n")
        f.write(f"Raw Reconstruction: {pred[0].cpu().min().item():.3f} to {pred[0].cpu().max().item():.3f}\n")
        f.write(f"Enhanced Output: {final_img.cpu().min().item():.3f} to {final_img.cpu().max().item():.3f}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"Raw PSNR: {psnr_raw:.2f} dB\n")
        f.write(f"Raw SSIM: {ssim_raw:.4f}\n")
        f.write(f"Enhanced PSNR: {psnr_enhanced:.2f} dB\n")
        f.write(f"Enhanced SSIM: {ssim_enhanced:.4f}\n")
        f.write(f"L2 Error (Raw): {l2_raw:.3f}\n")
        f.write(f"L2 Error (Enhanced): {l2_enhanced:.3f}\n\n")
        
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        f.write(f"Total training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"Total training time in seconds: {training_time:.2f}s\n")

    print(f"Saved metrics to: {metrics_path}")
    return psnr_raw, ssim_raw, psnr_enhanced, ssim_enhanced, l2_raw, l2_enhanced

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

    psnr_raw, ssim_raw, l2_raw = calculate_metrics(pred[0], x[0])
    psnr_enhanced, ssim_enhanced, l2_enhanced = calculate_metrics(final_img, x[0])
    
    print(f"\nMetrics:")
    print(f"Raw PSNR: {psnr_raw:.2f} dB, Raw SSIM: {ssim_raw:.4f}")
    print(f"Enhanced PSNR: {psnr_enhanced:.2f} dB, Enhanced SSIM: {ssim_enhanced:.4f}")
    print(f"L2 Error (Raw): {l2_raw:.3f}")
    print(f"L2 Error (Enhanced): {l2_enhanced:.3f}")

    # Visualization
    display_results(images, titles, save_dir)
    save_individual_images(images, titles, save_dir)

    # Save metrics to file
    save_metrics(y, x, pred, final_img, training_time, save_dir)

def main():
    """Main training and evaluation pipeline"""
    config, save_dir, mode = setup_experiment()
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Data loading
    x_train = np.load(config["data"]["x_path"])
    y_train = np.load(config["data"]["y_path"])
    x_train = prepare_tensor(x_train)[:config["data"]["subset_size"]]
    y_train = prepare_tensor(y_train)[:config["data"]["subset_size"]]

    # Convert grayscale to 3-channel if needed
    if x_train.shape[1] == 1:
        x_train = x_train.repeat(1, 3, 1, 1)
        y_train = y_train.repeat(1, 3, 1, 1)

    # Normalize
    x_train = x_train / x_train.max()
    y_train = y_train / y_train.max()
    
    # Dataset preparation
    full_dataset = DenoisingDataset(y_train, x_train)
    train_size = int(config["data"]["train_split"] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False
    )

    # Build model
    denoiser = dinv.models.DiffUNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    prior = PnP(denoiser=denoiser).to(device)
    physics = Denoising(
        noise_model=GaussianNoise(sigma=config["physics"]["sigma"])
    ).to(device)

    model = unfolded_builder(
        iteration="PGD",
        data_fidelity=L2(),
        max_iter=config["unfolded"]["max_iter"],
        prior=prior,
        trainable_params=config["unfolded"]["trainable_params"],
        params_algo={
            "stepsize": config["unfolded"]["stepsize"],
            "g_param": config["unfolded"]["g_param"],
            "beta": config["unfolded"]["beta"],
        },
    ).to(device)

    model.enable_checkpointing = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
    )

    loss = R2RLoss()

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=train_loader,
        epochs=config["training"]["epochs"],
        scheduler=scheduler,
        losses=loss,
        optimizer=optimizer,
        device=device,
        verbose=True,
        show_progress_bar=True,
        grad_clip=config["training"]["grad_clip"],
        wandb_vis=config["visualization"]["wandb_vis"],
        metrics=[SSIM(), PSNR()],
        plot_images=config["visualization"]["plot_images"],
        save_path=None,
        early_stop=config["training"]["early_stop"],
    )

    if mode == 'train':
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
        
        # Save the trained model
        model_path = os.path.join(save_dir, "trained_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Evaluate and save images after training
        print("Generating training results...")
        evaluate_model(model, physics, test_loader, device, training_time, save_dir)
        
    elif mode == 'test':
        # Load the trained model if it exists
        model_path = os.path.join(save_dir, "trained_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded trained model from: {model_path}")
        else:
            print("No trained model found. Please train first.")
            return
        
        # Create a smaller test loader for faster testing
        test_subset = torch.utils.data.Subset(test_dataset, indices=range(min(5, len(test_dataset))))
        fast_test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
        
        print(f"\nStarting testing with {len(test_subset)} samples (fast mode)")
        
        # Method 1: Try with custom metrics parameter
        print("=== TRAINER TEST WITH SSIM ===")
        test_metrics = trainer.test(
            fast_test_loader, 
            compare_no_learning=False, 
            verbose=True,
            metrics=[PSNR(), SSIM()]  # Explicitly specify metrics
        )
        
        # Generate and save images
        print("=== GENERATING VISUAL RESULTS ===")
        evaluate_model(model, physics, fast_test_loader, device, 0, save_dir)
        
if __name__ == "__main__":
    main()