import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import our dataset and transforms
from scripts.data_loader import StoneDataset, JointTransform
# Import models (uncomment the model you wish to use)
from models.deeplabv3plus import get_deeplabv3_plus
from models.segnet import get_segnet
from models.segformer import get_segformer_b5


def dice_coeff(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient for multi-class predictions using argmax.
    
    Args:
        pred (Tensor): Logits output from the model, shape [B, num_classes, H, W].
        target (Tensor): Ground truth mask with class labels, shape [B, H, W].
        smooth (float): Smoothing factor.
        
    Returns:
        float: Dice coefficient.
    """
    # Convert logits to probabilities, then to discrete predictions.
    pred = torch.softmax(pred, dim=1)
    pred = pred.argmax(dim=1)  # shape: [B, H, W]
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) score.
    
    Args:
        pred (Tensor): Logits output from the model, shape [B, num_classes, H, W].
        target (Tensor): Ground truth mask with class labels, shape [B, H, W].
        smooth (float): Smoothing factor.
        
    Returns:
        float: IoU score.
    """
    pred = torch.softmax(pred, dim=1)
    pred = pred.argmax(dim=1)  # shape: [B, H, W]
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def train_model(model, dataloaders, criterion, optimizer, device="cpu", num_epochs=10):
    """
    Train the model with the given parameters.
    
    Args:
        model (nn.Module): The segmentation model.
        dataloaders (dict): Dictionary with 'train' and 'val' DataLoaders.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').
        num_epochs (int): Number of training epochs.
        
    Returns:
        nn.Module: The trained model.
    """
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0.0
            running_iou = 0.0
            dataset_size = 0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # If the model returns a dict (e.g., for SegFormer),  we extract the logits.
                    if isinstance(outputs, dict) and "logits" in outputs:
                        outputs = outputs["logits"]

                    
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    loss = criterion(outputs, masks)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Calculate metrics.
                    dice = dice_coeff(outputs, masks)
                    iou = iou_score(outputs, masks)

                batch_size = inputs.size(0)
                dataset_size += batch_size
                running_loss += loss.item() * batch_size
                running_dice += dice.item() * batch_size
                running_iou += iou.item() * batch_size

            epoch_loss = running_loss / dataset_size
            epoch_dice = running_dice / dataset_size
            epoch_iou = running_iou / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Dice: {epoch_dice:.4f}  IoU: {epoch_iou:.4f}")

    return model


def main():
    # 1. Define paths to the augmented images and masks 
    image_dir = "data/augmented/images"
    mask_dir = "data/augmented/masks"

    # 2. Create the dataset using joint transformations.
    dataset = StoneDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=JointTransform(resize=(256, 256))
    )

    # 3. Split the dataset into training and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. Create DataLoaders for training and validation.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}

    # 5. Choose a model 
    # model = get_deeplabv3_plus(num_classes=2)
    # model = get_segnet(num_classes=2)
    model = get_segformer_b5(num_classes=2)

    # 6. Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 7. Train the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_model(model, dataloaders, criterion, optimizer, device=device, num_epochs=25)

    # 8. Save the trained model.
    save_path = "models/segformer_b5_stone.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

