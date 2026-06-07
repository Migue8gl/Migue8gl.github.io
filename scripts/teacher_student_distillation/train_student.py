import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


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


def get_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = CIFAR10("./data", train=True, download=True, transform=train_transform)
    test = CIFAR10("./data", train=False, download=True, transform=test_transform)

    n_eval = int(0.1 * len(full_train))
    train, eval_ = random_split(full_train, [len(full_train) - n_eval, n_eval])

    return (
        DataLoader(train, batch_size=128, shuffle=True),
        DataLoader(eval_, batch_size=128),
        DataLoader(test, batch_size=128),
    )


def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.5):
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    ce = F.cross_entropy(student_logits, labels)
    return alpha * ce + (1 - alpha) * (T * T) * kl


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train_baseline(model, train_loader, eval_loader, device, model_path):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=1e-5
    )
    early_stopping = EarlyStopping(patience=20)
    best_val_acc = 0.0

    for epoch in range(200):
        model.train()
        total, correct, n = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            total += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            n += bs
        scheduler.step()

        val_acc = accuracy(model, eval_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(
            f"[Baseline] epoch {epoch:3d} | "
            f"loss={total / n:.4f} | "
            f"train_acc={correct / n:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.5f}"
        )

        if early_stopping.step(val_acc):
            print(f"Early stopping en época {epoch}. Mejor val_acc: {best_val_acc:.4f}")
            break

    model.load_state_dict(torch.load(model_path))


def train_distill(student, teacher, train_loader, eval_loader, device, model_path):
    teacher.eval()
    opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=1e-5
    )
    early_stopping = EarlyStopping(patience=20)
    best_val_acc = 0.0

    for epoch in range(200):
        student.train()
        total, correct, n = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = y.size(0)
            total += loss.item() * bs
            correct += (s_logits.argmax(1) == y).sum().item()
            n += bs
        scheduler.step()

        val_acc = accuracy(student, eval_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), model_path)

        print(
            f"[Distill] epoch {epoch:3d} | "
            f"loss={total / n:.4f} | "
            f"train_acc={correct / n:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if early_stopping.step(val_acc):
            print(f"Early stopping en época {epoch}. Mejor val_acc: {best_val_acc:.4f}")
            break

    student.load_state_dict(torch.load(model_path))


def main():
    script_dir = get_script_dir()
    model_dir = os.path.join(script_dir, "models")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, eval_loader, test_loader = get_loaders()

    from torchvision.models import resnet50

    teacher = resnet50(weights=None)
    teacher.fc = nn.Linear(2048, 10)
    teacher.load_state_dict(
        torch.load(os.path.join(model_dir, "teacher.pth"), map_location=device)
    )
    teacher = teacher.to(device)

    student1 = StudentCNN().to(device)
    student2 = StudentCNN().to(device)

    train_baseline(
        student1,
        train_loader,
        eval_loader,
        device,
        os.path.join(model_dir, "student1_best.pth"),
    )
    train_distill(
        student2,
        teacher,
        train_loader,
        eval_loader,
        device,
        os.path.join(model_dir, "student2_best.pth"),
    )

    print("Baseline acc:", accuracy(student1, test_loader, device))
    print("Distill acc:", accuracy(student2, test_loader, device))


if __name__ == "__main__":
    main()
