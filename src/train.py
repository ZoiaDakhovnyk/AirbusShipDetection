import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset  # Import your custom dataset class
from model import UNet  # Import your model class
from utils.transforms import transform
from data import split_data, balance_dataset

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and loss function
model = UNet(in_channels=3, num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load processed data
X = np.load('C:/Users/Zoya/PycharmProjects/AirbusShipDetection/data/processed/X.npy')
y = np.load('C:/Users/Zoya/PycharmProjects/AirbusShipDetection/data/processed/y.npy')

# Split data
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data(X, y)

def main():

    # Balance the training dataset
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    # Define dataset and dataloaders
    train_dataset = CustomDataset(X_train_balanced, y_train_balanced, transform=transform)
    valid_dataset = CustomDataset(X_valid, y_valid, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)



    # Initialize the model, loss function, optimizer
    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            target = target.squeeze(1)
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device).unsqueeze(1)
                output = model(data)
                target = target.squeeze(1)
                loss = criterion(output, target.float())
                valid_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss/len(valid_loader)}")

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print("Model training completed and saved as 'model.pth'")

if __name__ == "__main__":
    main()