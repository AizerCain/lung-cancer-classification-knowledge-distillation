import torch
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from dataloader import get_dataloaders
from models import load_student


def main():
    # -------------------------
    # DEVICE
    # -------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # -------------------------
    # LOAD DATA
    # -------------------------
    _, _, test_loader = get_dataloaders(
        batch_size=16,
        num_workers=4   # SAFE because we are inside main()
    )

    # -------------------------
    # LOAD MODEL
    # -------------------------
    student = load_student()
    student.load_state_dict(
        torch.load("best_student_finetuned.pth", map_location=DEVICE)
    )
    student.to(DEVICE)
    student.eval()

    # -------------------------
    # TEST EVALUATION
    # -------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = student(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    print("\n📊 Classification Report\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=test_loader.dataset.classes
        )
    )

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_loader.dataset.classes,
        yticklabels=test_loader.dataset.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – Fine-Tuned GhostNet")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # INFERENCE SPEED (GPU)
    # -------------------------
    dummy_gpu = torch.randn(1, 3, 224, 224).to(DEVICE)

    # Warm-up
    for _ in range(10):
        _ = student(dummy_gpu)

    gpu_times = []
    for _ in range(100):
        start = time.time()
        _ = student(dummy_gpu)
        gpu_times.append(time.time() - start)

    print(f"\n🚀 GPU inference time: {np.mean(gpu_times)*1000:.2f} ms")

    # -------------------------
    # INFERENCE SPEED (CPU)
    # -------------------------
    student_cpu = load_student()
    student_cpu.load_state_dict(
        torch.load("best_student_finetuned.pth", map_location="cpu")
    )
    student_cpu.eval()

    dummy_cpu = torch.randn(1, 3, 224, 224)

    cpu_times = []
    for _ in range(100):
        start = time.time()
        _ = student_cpu(dummy_cpu)
        cpu_times.append(time.time() - start)

    print(f"💻 CPU inference time: {np.mean(cpu_times)*1000:.2f} ms")


# -------------------------
# WINDOWS ENTRY POINT
# -------------------------
if __name__ == "__main__":
    main()
