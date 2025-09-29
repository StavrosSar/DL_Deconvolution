import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import deepinv as dinv
from deepinv.loss.metric import SSIM, PSNR
from deepinv.physics import Denoising, GaussianNoise
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from skimage import exposure
import yaml
import os

# === CREATE SAVE DIRECTORY ===
save_dir = "../results/SCUNet_images_results"
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train denoising model')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--epochs', type=int, help='Override number of training epochs')
parser.add_argument('--batch_size', type=int, help='Override batch size')
parser.add_argument('--learning_rate', type=float, help='Override learning rate')
parser.add_argument('--sigma', type=float, help='Override noise sigma value')
parser.add_argument('--subset_size', type=int, help='Override dataset subset size')
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# === OVERRIDE CONFIG WITH COMMAND-LINE ARGUMENTS ===
if args.epochs is not None:
    config['training']['epochs'] = args.epochs
if args.batch_size is not None:
    config['data']['batch_size'] = args.batch_size
if args.learning_rate is not None:
    config['training']['learning_rate'] = args.learning_rate
if args.sigma is not None:
    config['physics']['sigma'] = args.sigma
if args.subset_size is not None:
    config['general']['subset_size'] = args.subset_size

# === DEVICE ===
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.cuda.empty_cache()

# === CUSTOM DATASET ===
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

# === LOAD DATA ===
x_train = np.load(config['data']['x_train_path'])
y_train = np.load(config['data']['y_train_path'])

subset_size = config['data']['subset_size']
x_train = prepare_tensor(x_train)[:subset_size]
y_train = prepare_tensor(y_train)[:subset_size]

# Convert grayscale to 3-channel if needed
if x_train.shape[1] == 1:
    x_train = x_train.repeat(1, 3, 1, 1)
    y_train = y_train.repeat(1, 3, 1, 1)

x_train = x_train / x_train.max()
y_train = y_train / y_train.max()

batch_size = config['training']['batch_size']
full_dataset = DenoisingDataset(y_train, x_train)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === MODEL & PHYSICS ===
model = dinv.models.SCUNet(pretrained=None, device=device).to(device)
physics = Denoising(noise_model=GaussianNoise(sigma=config['physics']['noise_sigma'])).to(device)

# === OPTIMIZER, SCHEDULER & LOSS ===
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
scheduler = StepLR(optimizer, step_size=int(config['training']['epochs']*0.8)+1)
loss_fn = dinv.loss.Neighbor2Neighbor()

model.enable_checkpointing = True

# === TRAINER ===
trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=config['training']['epochs'],
    scheduler=scheduler,
    losses=loss_fn,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    plot_images=True,
    show_progress_bar=True,
    verbose=True,
    wandb_vis=False,
    grad_clip=1.0,
    metrics=[SSIM(), PSNR()],
    save_path=None,
    early_stop=True
)

# === TRAINING ===
print(f"\nStarting training with {len(train_dataset)} samples")
start_time = time.time()
model = trainer.train()
end_time = time.time()

training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
print(f"Total training time: {hours}h {minutes}m {seconds}s")

# === EVALUATION & VISUALIZATION FUNCTIONS ===
def adaptive_normalize(tensor):
    normalized = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        img = tensor[i]
        q_low = torch.quantile(img, 0.01)
        q_high = torch.quantile(img, 0.99)
        normalized[i] = (img - q_low) / (q_high - q_low + 1e-8)
    return normalized.clamp(0, 1)

def contrast_stretch(img):
    img_np = img.cpu().numpy()
    p2, p98 = np.percentile(img_np, (2, 98))
    stretched = exposure.rescale_intensity(img_np, in_range=(p2, p98))
    return torch.from_numpy(stretched).to(img.device)

def prepare_for_display(img_tensor):
    img = img_tensor.squeeze().cpu().numpy()
    if len(img.shape) == 3:
        img = np.mean(img, axis=0)
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def display_inferno(images, titles, save_path=None):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        disp_img = prepare_for_display(img)
        im = ax.imshow(disp_img, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
        print(f"Saved image to: {save_path}")
    
    plt.show()

# === TEST & VISUALIZE ===
model.eval()
with torch.no_grad():
    y, x = next(iter(test_loader))
    y, x = y.to(device), x.to(device)
    pred = model(y, physics)
    pred_normalized = adaptive_normalize(pred)
    final_img = contrast_stretch(pred_normalized[0].unsqueeze(0))[0]

images = [y[0].cpu(), x[0].cpu(), pred[0].cpu(), final_img.cpu()]
titles = ["Noisy Input", "Ground Truth", "Raw Reconstruction", "Enhanced Reconstruction"]

# Save the comparison plot
comparison_path = os.path.join(save_dir, "comparison_plot.png")
display_inferno(images, titles, save_path=comparison_path)

# Save individual images
for i, (img, title) in enumerate(zip(images, titles)):
    disp_img = prepare_for_display(img)
    
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    filepath = os.path.join(save_dir, filename)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(disp_img, cmap='inferno', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    print(f"Saved {title} to: {filepath}")

# === NUMERICAL OUTPUTS ===
flat_input = y[0].cpu().flatten().numpy()
flat_target = x[0].cpu().flatten().numpy()
flat_recon = pred[0].cpu().flatten().numpy()
flat_enhanced = final_img.cpu().flatten().numpy()

np.set_printoptions(precision=6, threshold=20, linewidth=100)
print("\nFlattened NumPy Arrays of Sample 0")
print("\nInput (Noisy):", flat_input[:20])
print("Ground Truth:", flat_target[:20])
print("Raw Reconstruction:", flat_recon[:20])
print("Enhanced Reconstruction:", flat_enhanced[:20])

print("\nValue Ranges:")
print(f"Noisy Input: {y[0].cpu().min().item():.3f} to {y[0].cpu().max().item():.3f}")
print(f"Ground Truth: {x[0].cpu().min().item():.3f} to {x[0].cpu().max().item():.3f}")
print(f"Raw Reconstruction: {pred[0].cpu().min().item():.3f} to {pred[0].cpu().max().item():.3f}")
print(f"Enhanced Output: {final_img.cpu().min().item():.3f} to {final_img.cpu().max().item():.3f}")

# === METRICS ===
def calculate_metrics(pred, target):
    pred = pred.float().cpu()
    target = target.float().cpu()
    mse = torch.mean((pred - target)**2)
    psnr = 10 * torch.log10(1 / mse)
    l2 = torch.norm(pred - target)
    return psnr.item(), l2.item()

psnr_raw, l2_raw = calculate_metrics(pred[0], x[0])
psnr_enhanced, l2_enhanced = calculate_metrics(final_img, x[0])

print(f"\nMetrics:")
print(f"Raw PSNR: {psnr_raw:.2f} dB")
print(f"Enhanced PSNR: {psnr_enhanced:.2f} dB")
print(f"L2 Error (Raw): {l2_raw:.3f}")
print(f"L2 Error (Enhanced): {l2_enhanced:.3f}")

# === SAVE METRICS TO A TEXT FILE ===
metrics_path = os.path.join(save_dir, "metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("SCUNet Denoising Results\n")
    f.write("=======================\n\n")
    f.write("Value Ranges:\n")
    f.write(f"Noisy Input: {y[0].cpu().min().item():.3f} to {y[0].cpu().max().item():.3f}\n")
    f.write(f"Ground Truth: {x[0].cpu().min().item():.3f} to {x[0].cpu().max().item():.3f}\n")
    f.write(f"Raw Reconstruction: {pred[0].cpu().min().item():.3f} to {pred[0].cpu().max().item():.3f}\n")
    f.write(f"Enhanced Output: {final_img.cpu().min().item():.3f} to {final_img.cpu().max().item():.3f}\n\n")
    
    f.write("Metrics:\n")
    f.write(f"Raw PSNR: {psnr_raw:.2f} dB\n")
    f.write(f"Enhanced PSNR: {psnr_enhanced:.2f} dB\n")
    f.write(f"L2 Error (Raw): {l2_raw:.3f}\n")
    f.write(f"L2 Error (Enhanced): {l2_enhanced:.3f}\n")

print(f"Saved metrics to: {metrics_path}")