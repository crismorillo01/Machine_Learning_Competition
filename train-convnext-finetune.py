import os
import json
import random
import warnings
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


data_folder = "/Users/cristinamorilloleal/Documents/Máster Data Science/Primer Curso/Segundo Cuatri/Introduction to ML/Competition Project/"
train_folder = os.path.join(data_folder, "train")
checkpoint_path = "checkpoints/convnext-base-finetuned.pth"

epochs = 1
batch_size = 8
learning_rate = 1e-4
weight_decay = 1e-4
val_split = 0.15
freeze_epochs = 1
label_smoothing = 0.1
min_images_per_class = 2
num_workers = 0
seed = 42

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def seed_everything(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def build_transforms():
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = weights.transforms()
    return train_transform, eval_transform


def collect_samples(train_folder, min_images_per_class):
    train_folder = Path(train_folder)
    if not train_folder.exists():
        raise FileNotFoundError(f"Training folder not found: {train_folder}")

    class_to_paths = {}
    for class_folder in sorted(path for path in train_folder.iterdir() if path.is_dir()):
        paths = sorted(
            path
            for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if len(paths) >= min_images_per_class:
            class_to_paths[class_folder.name] = paths

    if len(class_to_paths) < 2:
        raise ValueError("Need at least two identities after filtering.")

    class_to_idx = {class_name: index for index,
                    class_name in enumerate(class_to_paths)}
    return class_to_paths, class_to_idx


def split_samples(class_to_paths, class_to_idx, val_split, seed_value):
    rng = random.Random(seed_value)
    train_samples = []
    val_samples = []

    for class_name, paths in class_to_paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        val_count = max(1, int(round(len(paths) * val_split)))
        val_count = min(val_count, len(paths) - 1)

        label = class_to_idx[class_name]
        val_paths = paths[:val_count]
        train_paths = paths[val_count:]
        train_samples.extend((path, label) for path in train_paths)
        val_samples.extend((path, label) for path in val_paths)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def make_loaders():
    train_transform, eval_transform = build_transforms()
    class_to_paths, class_to_idx = collect_samples(
        train_folder, min_images_per_class)
    train_samples, val_samples = split_samples(
        class_to_paths, class_to_idx, val_split, seed)

    train_dataset = ImagePathDataset(train_samples, transform=train_transform)
    val_dataset = ImagePathDataset(val_samples, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_to_idx


def build_model(num_classes):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return model.to(device)


def set_backbone_trainable(model, trainable):
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, training, phase_name):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
            print(
                f"{phase_name} batch {batch_idx}/{len(loader)} | "
                f"loss {total_loss / total:.4f} acc {correct / total:.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


def save_checkpoint(path, model, class_to_idx, epoch, best_val_acc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "convnext_base",
        "weights": "ConvNeXt_Base_Weights.IMAGENET1K_V1",
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "freeze_epochs": freeze_epochs,
        "label_smoothing": label_smoothing,
        "min_images_per_class": min_images_per_class,
    }
    torch.save(checkpoint, path)

    metadata_path = path.with_suffix(".json")
    metadata = {key: value for key, value in checkpoint.items()
                if key != "state_dict"}
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


seed_everything(seed)

print(f"Training folder: {train_folder}")
train_loader, val_loader, class_to_idx = make_loaders()
print(f"Number of identities: {len(class_to_idx)}")
print(f"Training images: {len(train_loader.dataset)}")
print(f"Validation images: {len(val_loader.dataset)}")

model = build_model(num_classes=len(class_to_idx))
criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_val_acc = 0.0
for epoch in range(1, epochs + 1):
    set_backbone_trainable(model, trainable=epoch > freeze_epochs)

    train_loss, train_acc = run_epoch(
        model, train_loader, criterion, optimizer, training=True, phase_name="train"
    )
    val_loss, val_acc = run_epoch(
        model, val_loader, criterion, optimizer, training=False, phase_name="val"
    )
    scheduler.step()

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f}"
    )

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(checkpoint_path, model,
                        class_to_idx, epoch, best_val_acc)
        print(f"Saved best checkpoint to {checkpoint_path}")

print(f"Best validation accuracy: {best_val_acc:.4f}")
