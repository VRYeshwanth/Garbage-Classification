import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from gradcam_utils import GradCAMExplainer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH = "./models/ResNet_FT.pt"
TEST_DATA_PATH = "./dataset/test"
OUTPUT_DIR = Path("./outputs/gradcam")

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Choose: "misclassified", "high_confidence_errors",
#         "class_pair", or "correct"
ANALYSIS_MODE = "misclassified"

# Used only for class_pair mode.
TRUE_CLASS = "glass"
PREDICTED_CLASS = "plastic"

# Number of images to explain for the selected mode.
NUM_IMAGES = 10

# Maximum number of examples to print in the summary.
SUMMARY_LIMIT = 20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", device)


# --------------------------------------------------
# 2. Load test dataset
# --------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = ImageFolder(
    TEST_DATA_PATH,
    transform=test_transform
)

class_names = test_dataset.classes
num_classes = len(class_names)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", class_names)
print("Test images:", len(test_dataset))


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Model loaded.")


# --------------------------------------------------
# 4. Predict the entire test dataset
# --------------------------------------------------

all_predictions = []
all_labels = []
all_confidences = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confidences.extend(confidences.cpu().tolist())


# --------------------------------------------------
# 5. Build prediction records
# --------------------------------------------------

records = []

for index, (true_idx, pred_idx, confidence) in enumerate(
    zip(all_labels, all_predictions, all_confidences)
):
    records.append({
        "index": index,
        "path": test_dataset.samples[index][0],
        "true_idx": true_idx,
        "pred_idx": pred_idx,
        "true_class": class_names[true_idx],
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "correct": true_idx == pred_idx
    })

correct_records = [
    record for record in records if record["correct"]
]

error_records = [
    record for record in records if not record["correct"]
]

accuracy = len(correct_records) / len(records)

print("\nTest-set summary")
print("----------------")
print(f"Total images: {len(records)}")
print(f"Correct: {len(correct_records)}")
print(f"Incorrect: {len(error_records)}")
print(f"Accuracy: {accuracy:.4%}")


# --------------------------------------------------
# 6. Save a complete prediction report
# --------------------------------------------------

report_path = OUTPUT_DIR / "test_predictions.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "index",
        "path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for record in records:
        writer.writerow({
            "index": record["index"],
            "path": record["path"],
            "true_class": record["true_class"],
            "predicted_class": record["predicted_class"],
            "confidence": f"{record['confidence']:.6f}",
            "correct": record["correct"]
        })

print("Prediction report saved to:", report_path)


# --------------------------------------------------
# 7. Select images for Grad-CAM
# --------------------------------------------------

if ANALYSIS_MODE == "misclassified":

    # Inspect errors, starting with the most confident mistakes.
    selected_records = sorted(
        error_records,
        key=lambda record: record["confidence"],
        reverse=True
    )[:NUM_IMAGES]


elif ANALYSIS_MODE == "high_confidence_errors":

    selected_records = sorted(
        error_records,
        key=lambda record: record["confidence"],
        reverse=True
    )[:NUM_IMAGES]


elif ANALYSIS_MODE == "class_pair":

    if TRUE_CLASS not in class_names:
        raise ValueError(f"Unknown true class: {TRUE_CLASS}")

    if PREDICTED_CLASS not in class_names:
        raise ValueError(
            f"Unknown predicted class: {PREDICTED_CLASS}"
        )

    selected_records = [
        record for record in error_records
        if record["true_class"] == TRUE_CLASS
        and record["predicted_class"] == PREDICTED_CLASS
    ]

    selected_records = sorted(
        selected_records,
        key=lambda record: record["confidence"],
        reverse=True
    )[:NUM_IMAGES]


elif ANALYSIS_MODE == "correct":

    # Take up to NUM_IMAGES correct examples from each class.
    selected_records = []

    for class_name in class_names:
        class_records = [
            record for record in correct_records
            if record["true_class"] == class_name
        ]

        selected_records.extend(class_records[:NUM_IMAGES])


else:
    raise ValueError(
        "Invalid ANALYSIS_MODE. Choose misclassified, "
        "high_confidence_errors, class_pair, or correct."
    )


if not selected_records:
    print("\nNo images matched the selected criteria.")

    if ANALYSIS_MODE == "class_pair":
        print(
            f"No examples of {TRUE_CLASS} predicted as "
            f"{PREDICTED_CLASS} were found."
        )

    raise SystemExit(0)


print(f"\nSelected {len(selected_records)} images")

for record in selected_records[:SUMMARY_LIMIT]:
    print(
        f"Index {record['index']:4d} | "
        f"True: {record['true_class']:12s} | "
        f"Predicted: {record['predicted_class']:12s} | "
        f"Confidence: {record['confidence']:.2%}"
    )


# --------------------------------------------------
# 8. Initialize reusable Grad-CAM utility
# --------------------------------------------------

explainer = GradCAMExplainer(
    model=model,
    target_layers=[model.layer4[-1]],
    class_names=class_names,
    device=device
)


# --------------------------------------------------
# 9. Generate and save explanations
# --------------------------------------------------

for record in selected_records:

    index = record["index"]

    image_tensor, true_label = test_dataset[index]

    # Explain the model's predicted class.
    result = explainer.explain(image_tensor)

    print(
        f"\nImage {index}: "
        f"true={record['true_class']}, "
        f"predicted={result['predicted_name']}, "
        f"confidence={result['confidence']:.2%}"
    )

    # Make a unique directory for each example.
    image_dir = OUTPUT_DIR / (
        f"image_{index:04d}_"
        f"true_{record['true_class']}_"
        f"pred_{record['predicted_class']}"
    )

    image_dir.mkdir(parents=True, exist_ok=True)

    # Save original image, heatmap and overlay.
    import matplotlib.pyplot as plt

    plt.imsave(
        image_dir / "original.png",
        result["original"]
    )

    plt.imsave(
        image_dir / "heatmap.png",
        result["heatmap"],
        cmap="jet",
        vmin=0,
        vmax=1
    )

    plt.imsave(
        image_dir / "overlay.png",
        result["overlay"]
    )

    # Save a comparison figure as well.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(result["original"])
    axes[0].set_title(
        f"True: {record['true_class']}"
    )

    axes[1].imshow(
        result["heatmap"],
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(result["overlay"])
    axes[2].set_title(
        f"Predicted: {result['predicted_name']}\n"
        f"Confidence: {result['confidence']:.2%}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(
        image_dir / "comparison.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

print(f"\nGrad-CAM analysis complete. Results: {OUTPUT_DIR}")
