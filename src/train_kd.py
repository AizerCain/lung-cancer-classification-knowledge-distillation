import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader import get_dataloaders
from models import (
    load_teacher,
    load_student,
    freeze_student_backbone
)

# -------------------------
# DEVICE & HYPERPARAMETERS
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
FINE_TUNE_EPOCHS = 5

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5

TEMPERATURE = 4.0
ALPHA = 0.4

# -------------------------
# DISTILLATION LOSS
# -------------------------
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce(student_logits, labels)

        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# -------------------------
# ACCURACY
# -------------------------
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)

# -------------------------
# EVALUATION (VAL)
# -------------------------
def evaluate(model, loader):
    model.eval()
    total_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_acc += accuracy(outputs, labels)

    return total_acc / len(loader)

# -------------------------
# KD TRAINING
# -------------------------
def train_kd(train_loader, val_loader):
    teacher = load_teacher()
    student = load_student()

    freeze_student_backbone(student)

    criterion = DistillationLoss(TEMPERATURE, ALPHA)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=LEARNING_RATE
    )

    best_val_acc = 0.0

    print("\n🚀 Starting Knowledge Distillation Training\n")

    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0
        running_acc = 0.0

        for images, labels in tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(student_logits, labels)

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        val_acc = evaluate(student, val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), "best_student_kd.pth")
            print("💾 Saved best KD student")

    return student, train_loader, val_loader

# -------------------------
# FINE-TUNING
# -------------------------
def fine_tune_student(student, train_loader, val_loader):
    print("\n🔓 Starting Fine-Tuning (Unfreezing Backbone)\n")

    for param in student.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(student.parameters(), lr=FINE_TUNE_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(FINE_TUNE_EPOCHS):
        student.train()
        running_acc = 0.0

        for images, labels in tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{FINE_TUNE_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = student(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_acc += accuracy(outputs, labels)

        val_acc = evaluate(student, val_loader)
        print(f"Fine-Tune Epoch {epoch+1} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), "best_student_finetuned.pth")
            print("💾 Saved best fine-tuned student")

# -------------------------
# ENTRY POINT (WINDOWS SAFE)
# -------------------------
if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    student, train_loader, val_loader = train_kd(train_loader, val_loader)
    fine_tune_student(student, train_loader, val_loader)

    print("\n✅ Training + Fine-Tuning Completed Successfully")
