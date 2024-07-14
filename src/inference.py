from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from model import UNet
from train import device, X_test, y_test
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from utils.transforms import transform

# Load the trained model
model = UNet(in_channels=3, num_classes=1).to(device)
# Ensure the model is loaded correctly on CPU if no CUDA is available
if not torch.cuda.is_available():
    model.load_state_dict(torch.load('C:/Users/Zoya/PycharmProjects/AirbusShipDetection/src/model.pth', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('C:/Users/Zoya/PycharmProjects/AirbusShipDetection/src/model.pth'))

criterion = nn.BCELoss()
#Define test dataset and test dataloader
test_dataset = CustomDataset(X_test, y_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
def dice_coefficient(pred, target, epsilon=1e-6):
    # Flatten the tensors
    pred = pred.view(-1).float()
    target = target.view(-1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()



def eval_loop(model, criterion, test_loader, device=device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        accuracy, f1_scores, dice_scores = [], [], []
        pbar = tqdm(test_loader, desc='Iterating over test data')
        for imgs, masks in pbar:
            # pass to device
            imgs = imgs.to(device)
            masks = masks.to(device)
            # forward
            out = model.forward(imgs)
            loss = criterion(out, masks)
            running_loss += loss.item() * imgs.shape[0]

            # calculate predictions using output
            predicted = (out > 0.3).float()
            predicted = predicted.view(-1).cpu().numpy()
            labels = masks.view(-1).cpu().numpy()

            # Convert labels to binary (0 or 1)
            labels = (labels > 0.5).astype(int)

            accuracy.append(accuracy_score(labels, predicted))
            f1_scores.append(f1_score(labels, predicted, average='binary'))

            # Calculate Dice coefficient
            dice = dice_coefficient(torch.tensor(predicted), torch.tensor(labels))
            dice_scores.append(dice)
    acc = sum(accuracy) / len(accuracy)
    f1 = sum(f1_scores) / len(f1_scores)
    dice = sum(dice_scores) / len(dice_scores)
    running_loss /= len(test_loader.sampler)

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'dice': dice,
        'loss': running_loss}


metrics = eval_loop(model, criterion, test_loader)
print('accuracy:', metrics['accuracy'])
print('f1 macro:', metrics['f1_macro'])
print('dice:', metrics['dice'])
print('test loss:', metrics['loss'])