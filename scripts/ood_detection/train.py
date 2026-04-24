import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_channels: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_channels),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    train_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=ToTensor()
    )

    num_classes = len(train_data.classes)

    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNet(in_channels=1, hidden_units=40, output_channels=num_classes)
    model = model.to(device)
    print(f"ConvNet size: {sum(p.numel() for p in model.parameters())}")

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        model.train()

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss_value = loss(y_pred, y)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()

            preds = y_pred.argmax(dim=1)
            train_acc += (preds == y).float().mean().item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print(f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    for X, y in test_dataloader:
        with torch.inference_mode():
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_value = loss(y_pred, y)
            test_loss += loss_value.item()

            preds = y_pred.argmax(dim=1)
            test_acc += (preds == y).float().mean().item()
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"Test loss={test_loss:.4f}, Test acc={test_acc:.4f}")

    models_path = os.path.join(script_dir, "models")
    os.makedirs(models_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_path, "convnet.pth"))


if __name__ == "__main__":
    main()
