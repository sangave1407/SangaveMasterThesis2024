import torch
from tqdm import tqdm 

class Trainer():
    def __init__(self, model, optimizer, criterion, device, result_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.last_validation_loss = float("inf")
        self.result_path = result_path

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.train(train_loader, epoch)
            self.validate(val_loader, epoch)

    def train(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        print(f"Running training for epoch {epoch}")
        for idx, (data, labels) in tqdm(enumerate(train_loader)):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if idx % 5 == 4:
                print("Epoch:", epoch + 1, "Batch:",
                      idx + 1, "Loss:", running_loss / 100)
                running_loss = 0.0

    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        print(f"Running validation for epoch {epoch}")
        with torch.no_grad():
            for idx, (data, labels) in tqdm(enumerate(val_loader)):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader)
        print(f"Validation loss: {val_loss}")

        if val_loss < self.last_validation_loss:
            self.last_validation_loss = val_loss
            path = f"{self.result_path}/model_{epoch}.pth"
            print("Validation loss decreased. Saving model.")
            self.save(path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
