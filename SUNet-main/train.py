import os
from this import d
import torch
import yaml
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Lambda
from torch.nn.functional import interpolate
import time
import utils
import numpy as np
import random
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
import gc

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

free_gpu_cache() 

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
SUNet = opt['SWINUNET']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)

#####EXAMPLEEEEEEEEEEEEEEE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_restored = model_restored.to(device)  # Move the model to the selected device


########################model_restored = model_restored.cuda()  # Move the model to GPU=model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
# utils.mkdir(model_dir)
# train_dir = Train['TRAIN_DIR']
# val_dir = Train['VAL_DIR']

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
'''
optimizer = optim.Adam(...): This line creates an instance of the Adam optimizer from PyTorch. Adam is a popular optimization
 algorithm that combines the advantages of two other extensions of stochastic gradient descent.
model_restored.parameters(): This passes the parameters of the model (presumably restored from a checkpoint or previously 
 trained model) to the optimizer so that it knows which parameters to optimize.
lr=new_lr: This sets the initial learning rate for the optimizer.
betas=(0.9, 0.999): These are the coefficients used for computing running averages of gradient and its square. 
 Common defaults for Adam are (0.9, 0.999).
eps=1e-8: A small constant added to improve numerical stability.
'''
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3


'''
scheduler = GradualWarmupScheduler(...): This sets up a gradual warmup scheduler that controls the learning rate during 
the initial warmup period and then switches to the cosine annealing scheduler for the remaining epochs.
multiplier=1: This could indicate that the learning rate does not change scaling during warmup.
total_epoch=warmup_epochs: This specifies how long the warmup phase lasts.

after_scheduler=scheduler_cosine: This links the cosine annealing scheduler to follow after the warmup period.
scheduler.step(): This call updates the learning rate scheduler to begin the first step of the learning rate adjustment. 

It typically needs to be called in the training loop after each training epoch to update the learning rate accordingly
'''

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest_ep-400_bs-16_ps-1.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
L1_loss = nn.L1Loss()

## DataLoaders
print('==> Loading datasets')
# train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
# train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
#                           shuffle=True, num_workers=0, drop_last=False)
# val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
#                         drop_last=False)

# Read Saved Batches   
x_train = np.load('C:/Users/user/Desktop/tr/x_train.npy')
y_train = np.load('C:/Users/user/Desktop/tr/y_train.npy')

# Normalize targets
y_train = y_train - np.mean(y_train, axis=(1,2), keepdims=True)
norm_fact = np.max(y_train, axis=(1,2), keepdims=True) 
y_train /= norm_fact

# Normalize & scale tikho inputs
x_train = x_train - np.mean(x_train, axis=(1,2), keepdims=True)
x_train /= norm_fact

# NCHW convention
x_train = np.moveaxis(x_train, -1, 1)
y_train = np.moveaxis(y_train, -1, 1)

# Convert to torch tensor
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
print(x_train.size(), y_train.size())

free_gpu_cache() 



from torchvision.transforms import Resize

# Define the transform to resize the images to 256x256
resize_transform = Resize((256, 256))

# Data Augmentation function with resizing
def augmentation(im, seed):
    random.seed(seed)
    a = random.choice([0,1,2,3])
    if a == 0:
        return resize_transform(im)  # Apply resize directly
    elif a == 1:
        ch = random.choice([1, 2, 3])
        return resize_transform(torch.rot90(im, ch, [-2,-1]))
    elif a == 2:
        ch = random.choice([-2, -1])
        return resize_transform(torch.flip(im, [ch]))
    elif a == 3:
        ch1 = random.choice([10, -10])
        ch2 = random.choice([-2, -1])
        return resize_transform(torch.roll(im, ch1, dims=ch2))








'''
## Data Augmentation funciton
def augmentation(im, seed):
    random.seed(seed)
    a = random.choice([0,1,2,3])
    if a==0:
        return im
    elif a==1:
        ch = random.choice([1, 2, 3])
        return torch.rot90(im, ch, [-2,-1])
    elif a==2:
        ch = random.choice([-2, -1])
        return torch.flip(im, [ch])
    elif a==3:
        ch1 = random.choice([10, -10])
        ch2 = random.choice([-2, -1])
        return torch.roll(im, ch1, dims=ch2)
'''

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    # Randomly split train-validation set for every epoch
    n_obj = x_train.size()[0] 
    n_train = np.int16(0.9*n_obj)

    ind = np.arange(n_obj)
    np.random.shuffle(ind)

    #we take as a tensor dataset the first 18076 of the train and then the test 2009
    train_dataset = TensorDataset(y_train[ind][:n_train], x_train[ind][:n_train])
    val_dataset = TensorDataset(y_train[ind][n_train:], x_train[ind][n_train:])


    #18076/4 = 4519 is the train_loader and the val_loader is still 2009 due to batch_size=1
    train_loader = DataLoader(dataset=train_dataset, batch_size=4,
                            shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)

    del train_dataset, val_dataset
    free_gpu_cache() 
    import torch.nn.functional as F

    model_restored.train()
    # for i, data in enumerate(tqdm(train_loader), 0):
    for i, data in enumerate(train_loader, 0): 
        # Forward propagation
        #for param in model_restored.parameters():
        #   param.grad = None
        target = data[0].to(device)     #tensor [4,1,128,128]
        input_ = data[1].to(device)  #tensor [4,1,128,128]
        print(f'before data agumentation',target.size(), input_.size())


        if input_.size(2) != 256 or input_.size(3) != 256:
            input_ = F.interpolate(input_, size=(256, 256), mode='nearest-exact')
        if target.size(2) != 256 or target.size(3) != 256:
            target = F.interpolate(target, size=(256, 256), mode='nearest-exact')
        
        
        seed = random.randint(0,1000000)
        target = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(target)
        input_ = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(input_)
        restored = model_restored(input_)
        print(f'restores size=', restored.size())
       
        
        '''
        if restored.size() != target.size():
            restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')
        '''

          # Ensure to handle the expected dimensions
        if restored.dim() == 3:  # Only C, H, W (not N)
            restored = restored.unsqueeze(0)  # Add batch dimension if needed
        elif restored.dim() == 2:  # If only C, L
            restored = restored.unsqueeze(-1).unsqueeze(-1)  # Adding height and width dimensions if necessary
    
        print(f'restored size after potential adjustments: {restored.size()}')  # Debugging line
    
        # Use interpolate only if spatial sizes differ
        if restored.size(2) != target.size(2) or restored.size(3) != target.size(3):
            restored = interpolate(restored, size=(target.size(2), target.size(3)), mode='nearest-exact')



        # Compute loss
        # loss = Charbonnier_loss(restored, target)
        loss = L1_loss(restored, target)   
        #print(f'Loss: {loss.item()}')  # Print loss value for monitoring

        del target, input_, data
        free_gpu_cache() 

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)
            with torch.no_grad():
                restored = model_restored(input_)
                if restored.size() != target.size():
                    restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(restored, target))

            del target, input_, data_val
            free_gpu_cache() 

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE']))) 
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE'])))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE'])))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    del train_loader, val_loader, epoch_loss
    free_gpu_cache() 

writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} seconds'.format((total_finish_time)))
