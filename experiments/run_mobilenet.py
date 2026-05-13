import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms

from evaluation.plots import plot_confusion_matrix
from evaluation.plots import plot_training_curves
from models.mobilenet.model import MobileNetV2Lite


DATA_DIR = "data"
IMG_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 50
STAGE_1_EPOCHS = 5
STAGE_2_UNFREEZE_RATIO = 0.50
BEST_MODEL_PATH = "best_mobilenet.pth"
PLOTS_DIR = "logs/plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def build_stratified_splits(targets, train_ratio=0.7, val_ratio=0.15, seed=42):

    generator = torch.Generator().manual_seed(seed)
    class_to_indices = {}

    for index, target in enumerate(targets):
        class_to_indices.setdefault(target, []).append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    for indices in class_to_indices.values():

        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[index] for index in permutation]

        train_count = int(train_ratio * len(shuffled_indices))
        val_count = int(val_ratio * len(shuffled_indices))

        train_indices.extend(shuffled_indices[:train_count])
        val_indices.extend(shuffled_indices[train_count:train_count + val_count])
        test_indices.extend(shuffled_indices[train_count + val_count:])

    return train_indices, val_indices, test_indices


def build_class_weights(targets, train_indices, num_classes):

    class_counts = torch.zeros(num_classes, dtype=torch.float)

    for index in train_indices:
        class_counts[targets[index]] += 1

    return class_counts.sum() / (num_classes * class_counts)


def build_dataloaders():

    train_transform, eval_transform = build_transforms()

    train_base_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=train_transform
    )

    eval_base_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=eval_transform
    )

    train_indices, val_indices, test_indices = build_stratified_splits(
        train_base_dataset.targets
    )

    class_weights = build_class_weights(
        train_base_dataset.targets,
        train_indices,
        len(train_base_dataset.classes)
    )

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(eval_base_dataset, val_indices)
    test_dataset = Subset(eval_base_dataset, test_indices)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 4,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_base_dataset.classes,
        class_weights
    )


def build_stage_1_optimizer(model):

    model.freeze_backbone()

    return optim.Adam(
        model.backbone.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )


def build_stage_2_optimizer(model):

    model.unfreeze_last_layers(ratio=STAGE_2_UNFREEZE_RATIO)

    backbone_params = [
        param
        for param in model.backbone.features.parameters()
        if param.requires_grad
    ]

    classifier_params = [
        param
        for param in model.backbone.classifier.parameters()
        if param.requires_grad
    ]

    return optim.Adam(
        [
            {"params": backbone_params, "lr": 3e-5},
            {"params": classifier_params, "lr": 1e-4},
        ],
        weight_decay=1e-4
    )


def get_learning_rates(optimizer, stage):

    if stage == 1:
        return 0.0, optimizer.param_groups[0]["lr"]

    return (
        optimizer.param_groups[0]["lr"],
        optimizer.param_groups[1]["lr"]
    )


def format_lr(lr):

    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def train_one_epoch(model, train_loader, criterion, optimizer):

    model.train()
    model.freeze_batchnorm()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total


def evaluate(model, data_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_with_loss(model, data_loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(data_loader), correct / total


def predict_loader(model, data_loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def main():

    torch.backends.cudnn.benchmark = True

    print(f"\nUsing device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names, class_weights = (
        build_dataloaders()
    )

    print("\nClasses:")
    print(class_names)

    model = MobileNetV2Lite(
        num_classes=len(class_names)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )
    optimizer = build_stage_1_optimizer(model)

    best_val_acc = -1.0
    counter = 0
    patience = 15
    stage = 1
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(EPOCHS):

        if epoch == STAGE_1_EPOCHS:
            stage = 2
            optimizer = build_stage_2_optimizer(model)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = evaluate_with_loss(model, val_loader, criterion)
        backbone_lr, head_lr = get_learning_rates(optimizer, stage)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Stage {stage} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Backbone LR: {format_lr(backbone_lr)} | "
            f"Head LR: {format_lr(head_lr)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    plot_training_curves(
        history,
        title="MobileNetV2 Training Curves",
        save_path=f"{PLOTS_DIR}/mobilenet_training_curves.png"
    )

    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )

    y_test, y_pred = predict_loader(model, test_loader)
    test_acc = float((y_pred == y_test).mean())

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=class_names,
        title="MobileNetV2 Confusion Matrix",
        save_path=f"{PLOTS_DIR}/mobilenet_confusion_matrix.png"
    )

    print("\n====================================")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("====================================")


if __name__ == "__main__":
    main()
