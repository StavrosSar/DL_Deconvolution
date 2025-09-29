import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import deepinv as dinv
from deepinv.loss.metric import SSIM, PSNR
from deepinv.loss import R2RLoss
from deepinv.unfolded import unfolded_builder
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import Denoising, GaussianNoise
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from skimage import exposure
import yaml
import os


# =========================
# Configuration and Setup
# =========================
def setup_experiment():
    save_dir = "../results/PnPSCUNet_images_results"
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Train PnP-SCUNet denoising model')
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

    # Override config with CLI args
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

    return config, save_dir, args.mode


# =========================
# Dataset
# =========================
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


def load_and_preprocess_data(config):
    x_train = np.load(config['data']['x_train_path'])
    y_train = np.load(config['data']['y_train_path'])

    subset_size = config['data']['subset_size']
    x_train = prepare_tensor(x_train)[:subset_size]
    y_train = prepare_tensor(y_train)[:subset_size]

    if x_train.shape[1] == 1:  # grayscale → 3-channel
        x_train = x_train.repeat(1, 3, 1, 1)
        y_train = y_train.repeat(1, 3, 1, 1)

    x_train = x_train / x_train.max()
    y_train = y_train / y_train.max()
    return x_train, y_train


# =========================
# Visualization & Metrics
# =========================
def prepare_for_display(img_tensor):
    img = img_tensor.squeeze().cpu().numpy()
    if len(img.shape) == 3:
        img = np.mean(img, axis=0)
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def display_results(images, titles, save_dir, colormap='inferno'):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        disp_img = prepare_for_display(img)
        im = ax.imshow(disp_img, cmap=colormap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_individual_images(images, titles, save_dir, colormap='inferno'):
    for img, title in zip(images, titles):
        disp_img = prepare_for_display(img)
        filename = title.lower().replace(" ", "_") + ".png"
        path = os.path.join(save_dir, filename)
        plt.figure(figsize=(6, 6))
        plt.imshow(disp_img, cmap=colormap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(title)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def calculate_metrics(pred, target):
    pred = pred.float().cpu().unsqueeze(0)
    target = target.float().cpu().unsqueeze(0)
    psnr = PSNR()(pred, target).item()
    ssim = SSIM()(pred, target).item()
    l2 = torch.norm(pred - target).item()
    return psnr, ssim, l2


def save_metrics(save_dir, training_time=None, test_metrics=None, eval_metrics=None):
    """Save metrics to file with proper formatting"""
    path = os.path.join(save_dir, "metrics.txt")
    with open(path, 'w') as f:
        f.write("PnP-SCUNet Results\n")
        f.write("==================\n\n")

        if training_time is not None:
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            f.write(f"Training time: {hours}h {minutes}m {seconds}s\n\n")

        if test_metrics is not None:
            f.write("Test Metrics (Full Dataset):\n")
            f.write("-" * 30 + "\n")
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    f.write(f"{k}: {v:.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")

        if eval_metrics is not None:
            f.write("Sample Evaluation Metrics:\n")
            f.write("-" * 30 + "\n")
            psnr, ssim, l2 = eval_metrics
            f.write(f"PSNR: {psnr:.2f} dB\n")
            f.write(f"SSIM: {ssim:.4f}\n")
            f.write(f"L2 Error: {l2:.3f}\n")


def evaluate_model(model, physics, test_loader, device, save_dir):
    """Evaluate model and save results"""
    model.eval()
    with torch.no_grad():
        y, x = next(iter(test_loader))
        y, x = y.to(device), x.to(device)
        pred = model(y, physics)

    # Calculate metrics for the sample
    psnr, ssim, l2 = calculate_metrics(pred[0], x[0])
    
    # Save images
    images = [y[0].cpu(), x[0].cpu(), pred[0].cpu()]
    titles = ["Noisy Input", "Ground Truth", "Reconstruction"]
    
    display_results(images, titles, save_dir)
    save_individual_images(images, titles, save_dir)
    
    return psnr, ssim, l2


# =========================
# Main
# =========================
def main():
    config, save_dir, mode = setup_experiment()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    x_train, y_train = load_and_preprocess_data(config)
    dataset = DenoisingDataset(y_train, x_train)

    train_size = int(config['train']['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False)

    # Model & physics
    denoiser = dinv.models.SCUNet(pretrained=None, device=device)
    prior = PnP(denoiser=denoiser).to(device)
    physics = Denoising(noise_model=GaussianNoise(sigma=config['physics']['sigma'])).to(device)

    model = unfolded_builder(
        iteration="PGD",
        data_fidelity=L2(),
        max_iter=config['algo']['max_iter'],
        prior=prior,
        trainable_params=config['algo']['trainable_params'],
        params_algo=config['algo']['params_algo'],
    ).to(device)

    model.enable_checkpointing = True

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['train']['learning_rate'],
                                  weight_decay=config['train']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config['train']['epochs'],
                                  eta_min=config['train']['eta_min'])
    loss = R2RLoss()
    
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        epochs=config['train']['epochs'],
        scheduler=scheduler,
        losses=loss,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_loader,
        metrics=[SSIM(), PSNR()],
        show_progress_bar=True,
        verbose=True,
        grad_clip=config['train']['grad_clip'],
        save_path=None,
        early_stop=True,
    )
    
    training_time = None
    test_metrics = None
    eval_metrics = None
    
    if mode == 'train':
        print(f"\nStarting training with {len(train_ds)} samples...")
        start_time = time.time()
        model = trainer.train()
        end_time = time.time()
        training_time = end_time - start_time
    
        # Print training time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        
        model_path = os.path.join(save_dir, "pnp_scunet_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Evaluate on a sample after training
        print("Evaluating on test sample...")
        eval_metrics = evaluate_model(model, physics, test_loader, device, save_dir)
        
        # Save metrics with training time and sample evaluation
        save_metrics(save_dir=save_dir, training_time=training_time, eval_metrics=eval_metrics)
        print("Training completed and results saved!")
        
    elif mode == 'test':
        model_path = os.path.join(save_dir, "pnp_scunet_model.pth")
        if not os.path.exists(model_path):
            print("No trained model found. Please train first.")
            return
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
        # Run full test evaluation
        test_metrics = trainer.test(test_loader)
        print("Test metrics:", test_metrics)
        
        # Evaluate on a sample for visualization
        eval_metrics = evaluate_model(model, physics, test_loader, device, save_dir)
        
        # Save metrics with test results
        save_metrics(save_dir=save_dir, test_metrics=test_metrics, eval_metrics=eval_metrics)
        print("Testing completed and results saved!")


if __name__ == "__main__":
    main()