import os
import torch
from monai.metrics import HausdorffDistanceMetric
import random
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, EnsureTyped, EnsureChannelFirstd,
    Orientationd, SpatialPadd, SpatialCropd
)
from monai.data import DataLoader, Dataset, partition_dataset, SmartCacheDataset
from monai.config import print_config
from model_segmamba.segmamba import SegMamba
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR

hausdorff_distance = HausdorffDistanceMetric(percentile=95, include_background=False)
hausdorff_distance_val = HausdorffDistanceMetric(percentile=95, include_background=False)


root_dir = '/datasets/tdt4265/mic/asoca'
num_epochs = 250
batch_size = 1
learning_rate = 0.001

def dice_coefficient(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()  
    intersection = (preds * targets).sum(dim=[2, 3, 4])  
    union = preds.sum(dim=[2, 3, 4]) + targets.sum(dim=[2, 3, 4])  
    dice = (2. * intersection + smooth) / (union + smooth) 
    return dice.mean()  


transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(512, 512, 224), mode='constant'),
    #SpatialCropd(keys=["image", "label"], roi_center=(256, 256, 112), roi_size=(512, 512, 112)),
    EnsureTyped(keys=["image", "label"]),
])


def compute_probability_map(label_data, crop_size):
    label_data = label_data.squeeze()
    z, y, x = label_data.shape
    #print(label_data.shape)
    
    depth, height, width = crop_size
    probability_map = np.zeros((z - depth + 1, y - height + 1, x - width + 1))

    #power_factor = 2
    scaling_factor = 0.1

    for start_z in range(0, z - depth + 1, 50):
        #print(start_z)
        for start_y in range(0, y - height + 1, 50):
            for start_x in range(0, x - width + 1, 14):
                crop = label_data[start_z:start_z + depth, start_y:start_y + height, start_x:start_x + width]
                #probability_map[start_z, start_y, start_x] = np.exp(np.sum(crop) / scaling_factor)
                #probability_map[start_z, start_y, start_x] = (np.sum(crop) ** power_factor)
                probability_map[start_z, start_y, start_x] = np.sum(crop)

    probability_map /= np.sum(probability_map)  
    return probability_map

def weighted_random_crop(probability_map, crop_size):
    z, y, x = probability_map.shape
    depth, height, width = crop_size
    
    flat_index = np.random.choice(a=z * y * x, p=probability_map.ravel())
    start_z = flat_index // (y * x)
    start_y = (flat_index % (y * x)) // x
    start_x = (flat_index % (y * x)) % x

    return start_z, start_y, start_x

def split_data(data, crop_size=(224, 224, 96)):
    parts = []
    num_subvolumes = 12
    label_data = data['label'] 
    #print(data['image'].shape)

    probability_map = compute_probability_map(label_data, crop_size)

    for _ in range(num_subvolumes):
        start_x, start_y, start_z = weighted_random_crop(probability_map, crop_size)
        #print('startsss', start_z, start_y, start_x)


        crop_transform = SpatialCropd(
            keys=["image", "label"],
            roi_start=[start_x, start_y, start_z],
            roi_end=[start_x + 224, start_y + 224, start_z + 96]
        )

        lists = crop_transform(data)['image']
        #print(lists.shape)

        parts.append(crop_transform(data))

    return parts



data_dicts_diseased_ctca = [{
    'image': os.path.join(root_dir, 'Diseased/CTCA', f'Diseased_{i}.nrrd'),
    'label': os.path.join(root_dir, 'Diseased/Annotations', f'Diseased_{i}.nrrd')
} for i in range(1, 20)]


data_dicts_diseased_normal = [{
    'image': os.path.join(root_dir, 'Normal/CTCA', f'Normal_{i}.nrrd'),
    'label': os.path.join(root_dir, 'Normal/Annotations', f'Normal_{i}.nrrd')
} for i in range(1, 20)]


data_dicts = data_dicts_diseased_ctca + data_dicts_diseased_normal

train_files, val_files = partition_dataset(data_dicts, ratios=[0.8, 0.2], shuffle=True)


model = SegMamba(in_chans=1, out_chans=1, depths=[2,2,2,2], feat_size=[48, 144, 192, 768], hidden_size=1024).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.92) 

scaler = GradScaler()


writer = SummaryWriter()


class PartsDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, transforms):
        self.data_files = data_files
        self.transforms = transforms

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_dict = self.data_files[idx]
        data = self.transforms(data_dict) 
        parts = split_data(data) 
        return {"parts": parts}

dataset = PartsDataset(train_files, transforms)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

val_data = PartsDataset(val_files, transforms)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Compute binary focal loss between target and output logits."""
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    targets = targets.float()
    at = alpha * targets + (1 - alpha) * (1 - targets)
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1-pt)**gamma * BCE_loss
    return F_loss.mean()

def combined_loss(outputs, labels, weight_dice=0.75, weight_focal=0.25):
    dice_loss = 1 - dice_coefficient(outputs, labels)
    focal_loss_value = focal_loss(outputs, labels)  
    return weight_dice * dice_loss + weight_focal * focal_loss_value

def run_epoch(loader, model, optimizer, is_training):
    model.train() if is_training else model.eval()

    total_epoch_loss = 0.0
    total_epoch_dice = 0.0
    total_parts_count = 0
    #total_epoch_hd95 = 0


    for batch_index, batch in enumerate(loader): 
        optimizer.zero_grad()
        total_batch_loss = 0.0
        total_batch_dice = 0.0
        part_count = 0
        part_num = 0

        parts = batch['parts']
     
    
        for part in parts:
            part_num += 1
            #print(part_num)
            #print(part)
            images, labels = part['image'].cuda(), part['label'].cuda()

            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, labels)
                dice_score = dice_coefficient(outputs, labels)
                hd95_score = hausdorff_distance(y_pred=outputs.int(), y=labels.int())


            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()

            total_batch_loss += loss.item()
            total_batch_dice += dice_score.item()
            #total_epoch_hd95 += hausdorff_distance(preds, labels)

            #print('raw_batch', dice_score.item()) to inspect the dice for each batch
            part_count += 1


        if is_training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #if nan is a problem then use this
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad()

        total_epoch_loss += total_batch_loss
        total_epoch_dice += total_batch_dice
        total_parts_count += part_count


        if part_count > 0:
            average_batch_loss = total_batch_loss / part_count
            average_batch_dice = total_batch_dice / part_count
            print(f"Batch {batch_index + 1}: Average Loss = {average_batch_loss:.4f}, Average Dice Coefficient = {average_batch_dice:.4f}")
    if total_parts_count > 0:
        average_epoch_loss = total_epoch_loss / total_parts_count
        average_epoch_dice = total_epoch_dice / total_parts_count
        #average_epoch_hd95 = total_epoch_hd95 / total_parts_count
    else:
        average_epoch_loss = 0.0
        average_epoch_dice = 0.0
    hd95_score = hausdorff_distance.aggregate().item()
    hausdorff_distance.reset()


    return average_epoch_loss, average_epoch_dice, hd95_score

def validate(loader, model):
    model.eval() 
    
    total_val_loss = 0.0
    total_val_dice = 0.0
    total_parts_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            total_batch_loss = 0.0
            total_batch_dice = 0.0
            total_batch_hd95 = 0.0
            part_count = 0

            parts = batch['parts'] 
            for part in parts:
                images, labels = part['image'].cuda(), part['label'].cuda()

                outputs = model(images)
                loss = combined_loss(outputs, labels)
                dice_score = dice_coefficient(outputs, labels)
                hd95_score = hausdorff_distance_val(y_pred=outputs.int(), y=labels.int())

                total_batch_loss += loss.item()
                total_batch_dice += dice_score.item()
                #total_batch_hd95 += hd95_score.item()
                part_count += 1

            total_val_loss += total_batch_loss
            total_val_dice += total_batch_dice
            #total_val_hd95 += total_batch_hd95
            total_parts_count += part_count

        
    if total_parts_count > 0:
        average_val_loss = total_val_loss / total_parts_count
        average_val_dice = total_val_dice / total_parts_count
    else:
        average_val_loss = 0.0
        average_val_dice = 0.0

    #print(f'Validation: Average Loss = {average_val_loss:.4f}, Average Dice Coefficient = {average_val_dice:.4f}, Average HD95 = {average_val_hd95:.4f}')

    hd95_score = hausdorff_distance_val.aggregate().item()
    hausdorff_distance_val.reset()

    return average_val_loss, average_val_dice, hd95_score



epoch_dice_loss_list = []
train_loss_list = []
epoch_dice_val = []
epoch_hd95 = []
for epoch in range(num_epochs):
    train_loss, train_dice, hd95 = run_epoch(train_loader, model, optimizer, is_training=True)
    #print(train_loss)
    print('/n/n/n')
    #print(train_dice)
    epoch_dice_loss_list.append(train_dice)
    train_loss_list.append(train_loss)
    if epoch%5==0:
        val_loss, val_dice, val_hd95 = validate(val_loader, model)
    print(f'val_loss: {val_loss}, val_dice: {val_dice}, val_hd95: {val_hd95}')

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('DiceCoefficient/Train', train_dice, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Dice Coefficient: {train_dice}. HD95: {hd95}')
    scheduler.step()
    epoch_dice_val.append(val_dice)
    epoch_hd95.append(val_hd95)


    #os.makedirs("modelos", exist_ok=True)


    if epoch == num_epochs-1:
        torch.save(model.state_dict(), os.path.join("modelos", f'latest_model.pth'))
    if val_dice == max(epoch_dice_val):
        torch.save(model.state_dict(), os.path.join("modelos", f'best_epoch_{epoch+1}_model.pth'))

writer.close()
print('Finished Training')


