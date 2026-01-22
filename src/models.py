import torch
import torch.nn as nn
import timm

# -------------------------
# DEVICE
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.backends.cudnn.benchmark = True

NUM_CLASSES = 3

# -------------------------
# TEACHER MODEL (EdgeNeXt)
# -------------------------
def load_teacher():
    teacher = timm.create_model(
        "edgenext_small",
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    teacher = teacher.to(DEVICE)
    teacher.eval()  # Teacher is not trained

    return teacher

# -------------------------
# STUDENT MODEL (GhostNet)
# -------------------------
def load_student():
    student = timm.create_model(
        "ghostnet_100",
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    student = student.to(DEVICE)
    student.train()

    return student

# -------------------------
# FREEZE STUDENT BACKBONE
# -------------------------
def freeze_student_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

# -------------------------
# SANITY CHECK
# -------------------------
if __name__ == "__main__":
    teacher = load_teacher()
    student = load_student()

    freeze_student_backbone(student)

    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        teacher_out = teacher(dummy_input)

    student_out = student(dummy_input)

    print("Teacher output shape:", teacher_out.shape)
    print("Student output shape:", student_out.shape)

    print("✅ Models loaded and working correctly")
