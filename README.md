# ♻️ Smart Waste Classification Using Deep Learning

An end-to-end deep learning project that classifies waste images into 10 categories using **PyTorch, ResNet18 transfer learning, and Grad-CAM**. The project includes systematic fine-tuning experiments, model evaluation, and an interactive Streamlit web application for waste classification and model interpretability.

## ✨ Features

- **10-Class Waste Classification:** Classifies images into battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash.
- **Transfer Learning:** Uses a pretrained ResNet18 architecture with progressive fine-tuning of deeper layers.
- **Experiment Comparison:** Evaluates different fine-tuning strategies to study their impact on validation performance.
- **Model Evaluation:** Uses accuracy, precision, recall, F1-score, and confusion matrices to assess classification performance.
- **Grad-CAM Visualization:** Highlights image regions that influence the model's predictions.
- **Interactive Web Application:** Provides a Streamlit interface for uploading images, viewing predictions, and exploring model performance.
- **Model Limitations:** Documents challenges involving unfamiliar objects, background variation, packaging, and real-world images.

## 📊 Dataset

This project uses the [Garbage Classification V2 dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) from Kaggle, using the 384 standardized version.

The dataset contains 12,259 images across 10 classes.

| Class | Number of Images |
|---|---:|
| Battery | 756 |
| Biological | 699 |
| Cardboard | 1,411 |
| Clothes | 1,892 |
| Glass | 1,736 |
| Metal | 930 |
| Paper | 1,336 |
| Plastic | 1,597 |
| Shoes | 1,449 |
| Trash | 453 |
| **Total** | **12,259** |

### 📂 Dataset Split

The dataset was divided into training, validation, and test sets using a 70:20:10 split for each class.

- **Training set (70%):** Used to train the model.
- **Validation set (20%):** Used to compare experiments and select the best checkpoint.
- **Test set (10%):** Used for final evaluation.

The test set is kept separate from the training and validation process.

## 🧠 Model Architecture

The project uses **ResNet18**, a convolutional neural network pretrained on ImageNet, as the base architecture.

The original classification layer was replaced with a fully connected layer for the 10 waste categories.

### 🔬 Transfer Learning Strategy

Three experiments were conducted to evaluate the effect of fine-tuning different parts of the network.

| Experiment | Trainable Components | Best Validation Accuracy |
|---|---|---:|
| Experiment 1 | Final fully connected layer (`fc`) | 85.57% |
| Experiment 2 | `fc` and `layer4` | 91.06% |
| Experiment 3 | `fc`, `layer4`, and `layer3` | 92.48% |

Experiment 3 achieved the highest reported validation accuracy among these experiments.

### ⚙️ Training Configuration

- **Architecture:** ResNet18
- **Framework:** PyTorch
- **Optimizer:** Adam
- **Weight decay:** 0.0001
- **Classifier learning rate:** 0.0001
- **Layer 4 learning rate:** 0.00005
- **Layer 3 learning rate:** 0.00001
- **Input resolution:** 224 × 224 pixels
- **Normalization:** ImageNet mean and standard deviation
- **Checkpoint selection:** Best validation accuracy

### 🖼️ Image Preprocessing

Images are resized to 224 × 224 pixels and normalized using ImageNet statistics.

Training augmentations include horizontal flipping, vertical flipping, and small rotations. Evaluation uses the corresponding resizing and normalization steps without random augmentation.

## 📈 Model Evaluation

The model is evaluated using multiple metrics to provide a more complete assessment than accuracy alone.

- **Accuracy:** Proportion of correctly classified images.
- **Precision:** Measures how often predictions for a class are correct.
- **Recall:** Measures how many actual instances of a class are identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct predictions and misclassifications across the 10 classes.

The project distinguishes validation performance from final test-set performance. Validation results are used for experiment comparison, while test metrics should be reported separately after evaluating the selected checkpoint.

## 🔍 Explainability with Grad-CAM

The project integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image regions that influence a model's classification.

Grad-CAM generates heatmaps that can be overlaid on input images, helping users inspect which regions contributed to a prediction.

This feature provides an additional interpretability tool alongside the predicted class and confidence score.

**Important:** Grad-CAM visualizations are approximate explanations of model behavior. They do not prove that the model is reasoning correctly or relying exclusively on relevant object features.

## 🌐 Streamlit Web Application

The application provides an interactive interface with three main sections.

### 1. Classify Waste

- Upload a waste image.
- Generate a predicted class and confidence score.
- View a Grad-CAM heatmap and overlay to inspect influential image regions.

### 2. Model Performance

- Review model evaluation metrics.
- Explore the confusion matrix and class-level performance.
- Inspect precomputed test predictions when the required prediction file is available.

### 3. About the Project

- Explore the project methodology and model details.
- Review the training experiments.
- Understand the limitations of the classifier.

## 🚀 Installation and Usage

### Prerequisites

- Python version compatible with the project's dependencies
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/VRYeshwanth/Garbage-Classification.git
cd Garbage-Classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

### 📄 Required Files

Ensure that the repository includes the following files:

- `models/ResNet_FT.pt` — trained ResNet18 checkpoint.
- `src/gradcam_utils.py` — Grad-CAM implementation.
- `app.py` — Streamlit application entry point.
- `requirements.txt` — application dependencies.

The dataset is not required to run inference if the trained checkpoint is included. The Model Performance page may also require `outputs/gradcam/test_predictions.csv`, depending on the implemented evaluation workflow.

## 📁 Project Structure

```text
Directory structure:
└── Garbage-Classification/
    ├── app.py
    ├── images/
    │   ├── Baseline - Accuracy.png
    │   ├── Baseline - Loss.png
    │   ├── Confusion Matrix FT (Test Dataset).png
    │   ├── FT - Accuracy.png
    │   ├── FT - Loss.png
    │   ├── Partial_FT - Accuracy.png
    │   └── Partial_FT - Loss.png
    ├── models/
    │   ├── ResNet_Frozen_Backbone.pt
    │   ├── ResNet_FT.pt
    │   └── ResNet_Partial_FT.pt
    ├── notebooks/
    │   ├── 1.Dataset_Splitting.ipynb
    │   ├── 2.ResNet Baseline.ipynb
    │   ├── 3.ResNet Partial FT.ipynb
    │   ├── 4.ResNet FT.ipynb
    │   └── 5.Evaluation.ipynb
    ├── outputs/
    │   └── gradcam/
    │       ├── image_0090_true_biological_pred_trash/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0617_true_glass_pred_metal/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0658_true_metal_pred_plastic/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0744_true_metal_pred_cardboard/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0756_true_paper_pred_trash/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0862_true_paper_pred_trash/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0876_true_paper_pred_cardboard/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0957_true_plastic_pred_paper/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_0966_true_plastic_pred_cardboard/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       ├── image_1152_true_shoes_pred_clothes/
    │       │   ├── comparison.png
    │       │   ├── heatmap.png
    │       │   ├── original.png
    │       │   └── overlay.png
    │       └── test_predictions.csv
    ├── README.md
    ├── requirements.txt
    └── src/
        ├── gradcam_utils.py
        └── test_gradcam.py
```

The structure above highlights the main application components. Additional files may exist within the project directories.

## ⚠️ Limitations

- **Dataset Diversity:** The dataset may not represent all real-world backgrounds, lighting conditions, object orientations, and packaging.
- **Fixed Categories:** The classifier supports only the 10 classes used during training.
- **No Unknown-Class Detection:** Unfamiliar objects may still be assigned to one of the supported categories.
- **Mixed Waste:** The model performs whole-image classification and is not designed to identify multiple objects or separate waste items within a single image.
- **Confidence Interpretation:** A high softmax confidence score does not guarantee that a prediction is correct.
- **Explainability:** Grad-CAM highlights influential regions but cannot independently establish that the model uses appropriate visual features.
- **Generalization:** Performance on unseen images may differ from the reported validation and test results.


## 🧰 Technologies Used

- Python
- PyTorch
- TorchVision
- ResNet18
- Transfer Learning
- Grad-CAM
- Streamlit
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## 📚 References

- [Garbage Classification V2 Dataset — Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)