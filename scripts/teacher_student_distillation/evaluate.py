import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50


class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def get_test_loader():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10("./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=256)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = get_test_loader()

    teacher = resnet50(weights=None)
    teacher.fc = nn.Linear(2048, 10)
    teacher.load_state_dict(
        torch.load(os.path.join(model_dir, "teacher.pth"), map_location=device)
    )

    student_base = StudentCNN()
    student_base.load_state_dict(
        torch.load(os.path.join(model_dir, "student1_best.pth"), map_location=device)
    )

    student_distil = StudentCNN()
    student_distil.load_state_dict(
        torch.load(os.path.join(model_dir, "student2_best.pth"), map_location=device)
    )

    models = {
        "Teacher (ResNet50)": teacher,
        "Student baseline": student_base,
        "Student distilled": student_distil,
    }

    teacher_params = count_params(teacher)

    print(
        f"\n{'Model':<25} {'Acc (test)':<12} {'Params':>12} {'Compression':>13} {'Efficiency':>12}"
    )
    print("─" * 78)
    for name, model in models.items():
        model = model.to(device)
        acc = test_accuracy(model, loader, device)
        params = count_params(model)
        compression = teacher_params / params
        efficiency = acc * compression
        print(
            f"{name:<25} {acc:<12.4f} {params:>12,} {compression:>12.1f}x {efficiency:>12.3f}"
        )


if __name__ == "__main__":
    main()
