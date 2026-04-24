import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from train import ConvNet


def compute_msp(model, loader, device):
    all_msp = []

    with torch.inference_mode():
        for X, _ in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            msp = probs.max(dim=1).values
            all_msp.append(msp.cpu())

    return torch.cat(all_msp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    model_path = os.path.join(script_dir, "models", "convnet.pth")
    fig_path = os.path.join(script_dir, "msp_ood_vs_id.png")

    id_test = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    ood_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        ),
    )

    num_classes = len(id_test.classes)
    model = ConvNet(in_channels=1, hidden_units=40, output_channels=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    id_loader = torch.utils.data.DataLoader(id_test, batch_size=64)
    ood_loader = torch.utils.data.DataLoader(ood_test, batch_size=64)

    msp_id = compute_msp(model, id_loader, device)
    msp_ood = compute_msp(model, ood_loader, device)

    scores = torch.cat([msp_id, msp_ood]).numpy()
    labels = torch.cat([torch.ones_like(msp_id), torch.zeros_like(msp_ood)]).numpy()

    auroc = roc_auc_score(labels, scores)

    print(f"AUROC: {auroc:.4f}")

    plt.figure(figsize=(8, 5))

    plt.hist(
        msp_id.numpy(), bins=50, alpha=0.6, label="ID (FashionMNIST)", density=True
    )
    plt.hist(msp_ood.numpy(), bins=50, alpha=0.6, label="OOD (CIFAR-10)", density=True)

    plt.axvline(msp_id.mean().item(), color="blue", linestyle="--", linewidth=1)
    plt.axvline(msp_ood.mean().item(), color="orange", linestyle="--", linewidth=1)

    plt.title(f"MSP Distribution (AUROC={auroc:.4f})")
    plt.xlabel("Maximum Softmax Probability (MSP)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Figure saved at: {fig_path}")

    y_true = torch.cat([torch.ones_like(msp_id), torch.zeros_like(msp_ood)]).numpy()

    y_score = torch.cat([msp_id, msp_ood]).numpy()

    display = RocCurveDisplay.from_predictions(
        y_true, y_score, name="MSP OOD detection", plot_chance_level=True
    )

    display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve (ID vs OOD using MSP)",
    )

    plt.tight_layout()

    out_path = os.path.join(script_dir, "roc_msp_ood.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved ROC curve to {out_path}")


if __name__ == "__main__":
    main()
