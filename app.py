
from pathlib import Path
import sys
import hashlib
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "ResNet_FT.pt"

PREDICTIONS_PATH = (
    BASE_DIR / "outputs" / "gradcam" / "test_predictions.csv"
)

IMAGE_SIZE = 224
NUM_CLASSES = 10

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Make gradcam_utils.py importable from src/
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gradcam_utils import GradCAMExplainer


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartSort",
    page_icon="♻️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Load model once and cache it
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource
def load_explainer():
    model = load_model()

    return GradCAMExplainer(
        model=model,
        target_layers=[model.layer4[-1]],
        class_names=CLASS_NAMES,
        device=DEVICE,
    )


# --------------------------------------------------
# 4. Image preprocessing
# --------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 5. Load saved test-set predictions
# --------------------------------------------------

@st.cache_data
def load_test_predictions():
    if not PREDICTIONS_PATH.exists():
        return None

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    }

    if not required_columns.issubset(df.columns):
        return None

    return df


# --------------------------------------------------
# 6. Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.title("♻️ SmartSort")
    st.caption("Smart Waste Classification")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Classify Waste",
            "Model Performance",
            "About the Project",
        ],
    )

    st.divider()
    st.caption(f"Device: {DEVICE.type.upper()}")
    st.caption("Model: ResNet18")


# --------------------------------------------------
# 7. Page: Classify Waste
# --------------------------------------------------

if page == "Classify Waste":

    st.title("Smart Waste Classification")

    st.write(
        "Upload an image of a waste item to classify it "
        "and visualize the model's Grad-CAM explanation."
    )

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"],
        key="waste_image_uploader",
    )

    # ----------------------------------------------
    # 7A. Detect the currently uploaded image
    # ----------------------------------------------

    if uploaded_file is not None:

        # Read the uploaded bytes and create a unique ID.
        uploaded_bytes = uploaded_file.getvalue()

        current_image_id = hashlib.sha256(
            uploaded_bytes
        ).hexdigest()

        # If the image changes, invalidate the old result.
        previous_image_id = st.session_state.get(
            "current_image_id"
        )

        if previous_image_id != current_image_id:

            st.session_state["current_image_id"] = (
                current_image_id
            )

            st.session_state.pop(
                "prediction_result",
                None,
            )

        try:
            original_image = Image.open(
                BytesIO(uploaded_bytes)
            ).convert("RGB")

            image_tensor = image_transform(
                original_image
            )

            st.subheader("Uploaded Image")

            st.image(
                original_image,
                caption="Input image",
                width=400,
            )

            # --------------------------------------
            # 7B. Run classification on button click
            # --------------------------------------

            if st.button(
                "Classify Image",
                type="primary",
                use_container_width=True,
                key="classify_image_button",
            ):

                with st.spinner(
                    "Classifying image and generating Grad-CAM..."
                ):

                    model = load_model()
                    explainer = load_explainer()

                    input_tensor = (
                        image_tensor.unsqueeze(0).to(DEVICE)
                    )

                    with torch.inference_mode():

                        logits = model(input_tensor)

                        probabilities = torch.softmax(
                            logits,
                            dim=1,
                        )[0]

                    predicted_idx = int(
                        probabilities.argmax().item()
                    )

                    confidence = float(
                        probabilities[predicted_idx].item()
                    )

                    # Generate Grad-CAM using the current image.
                    result = explainer.explain(
                        image_tensor,
                        original_image=original_image,
                    )

                    # Store the image ID with its prediction.
                    # This prevents results from one image
                    # being displayed for another image.
                    st.session_state["prediction_result"] = {
                        "image_id": current_image_id,
                        "image": original_image.copy(),
                        "predicted_idx": predicted_idx,
                        "confidence": confidence,
                        "probabilities": (
                            probabilities.cpu().tolist()
                        ),
                        "gradcam": result,
                    }

            # --------------------------------------
            # 7C. Display only the matching result
            # --------------------------------------

            saved = st.session_state.get(
                "prediction_result"
            )

            # The image IDs must match before results appear.
            if (
                saved is not None
                and saved.get("image_id") == current_image_id
            ):

                st.divider()
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Category",
                        CLASS_NAMES[
                            saved["predicted_idx"]
                        ].title(),
                    )

                with col2:
                    st.metric(
                        "Model Confidence",
                        f"{saved['confidence']:.2%}",
                    )

                st.caption(
                    "Confidence is the model's softmax score, "
                    "not a guarantee that the prediction is correct."
                )

                # ----------------------------------
                # Class probabilities
                # ----------------------------------

                st.subheader("Class Probabilities")

                probability_df = pd.DataFrame({
                    "Category": [
                        name.title()
                        for name in CLASS_NAMES
                    ],
                    "Probability": saved["probabilities"],
                }).sort_values(
                    "Probability",
                    ascending=False,
                )

                st.bar_chart(
                    probability_df.set_index("Category")
                )

                # ----------------------------------
                # Grad-CAM explanation
                # ----------------------------------

                st.subheader("Grad-CAM Explanation")

                st.write(
                    "The heatmap highlights image regions that "
                    "positively influence the selected class score. "
                    "It is an explanation aid, not proof of "
                    "correct reasoning."
                )

                cam_result = saved["gradcam"]

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        saved["image"],
                        caption="Original Image",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        cam_result["overlay"],
                        caption="Grad-CAM Overlay",
                        use_container_width=True,
                    )

        except Exception as error:

            st.error(
                "Unable to process this image. "
                "Check the model checkpoint and Grad-CAM setup."
            )

            st.exception(error)

    else:
        # No image is uploaded: remove any stale result.
        st.session_state.pop(
            "current_image_id",
            None,
        )

        st.session_state.pop(
            "prediction_result",
            None,
        )

        st.info(
            "Upload an image to get started. "
            "Your prediction will appear after you click "
            "'Classify Image'."
        )


# --------------------------------------------------
# 8. Page: Model Performance
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Evaluation results from the held-out test dataset."
    )

    df = load_test_predictions()

    if df is None:

        st.warning(
            "Test prediction report not found or invalid. "
            "Run your test evaluation script first to create "
            "outputs/gradcam/test_predictions.csv."
        )

    else:

        y_true = df["true_class"].astype(str)
        y_pred = df["predicted_class"].astype(str)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0,
            )
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Test Accuracy",
            f"{accuracy:.2%}",
        )

        col2.metric(
            "Macro Precision",
            f"{precision:.2%}",
        )

        col3.metric(
            "Macro Recall",
            f"{recall:.2%}",
        )

        col4.metric(
            "Macro F1-Score",
            f"{f1:.2%}",
        )

        st.caption(
            f"Evaluation uses {len(df):,} saved test predictions."
        )

        st.divider()
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        image = ax.imshow(
            cm,
            cmap="Greens",
        )

        fig.colorbar(
            image,
            ax=ax,
        )

        ax.set_xticks(
            range(len(CLASS_NAMES))
        )

        ax.set_yticks(
            range(len(CLASS_NAMES))
        )

        ax.set_xticklabels(
            CLASS_NAMES,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            CLASS_NAMES,
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Test Set Confusion Matrix")

        threshold = cm.max() / 2 if cm.size else 0

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):

                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[row, col] > threshold
                        else "black"
                    ),
                )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# --------------------------------------------------
# 9. Page: About the Project
# --------------------------------------------------

else:

    st.title("About SmartSort")

    st.write(
        """
        SmartSort is a computer vision project that classifies
        waste images into ten categories using a fine-tuned ResNet18
        model. It also uses Grad-CAM to visualize image regions
        that positively influence a selected prediction.
        """
    )

    st.subheader("Supported Categories")

    st.write(
        ", ".join(
            name.title()
            for name in CLASS_NAMES
        )
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Python** — application logic
        - **PyTorch and Torchvision** — model inference
        - **ResNet18** — image classification
        - **pytorch-grad-cam** — model explanations
        - **Streamlit** — web interface
        - **Pandas and scikit-learn** — evaluation metrics
        - **Matplotlib** — confusion matrix visualization
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - **Limited Dataset Diversity:** The model may struggle with
        variations in backgrounds, lighting, object orientations,
        packaging, and real-world environments.
        - **Fixed Categories:** Predictions are limited to ten
        predefined waste categories. The model cannot explicitly
        reject unfamiliar objects or assign multiple labels.
        - **Whole-Image Classification:** The model predicts one
        category for the entire image rather than identifying
        individual waste objects. Images containing multiple
        waste types may be misclassified.
        - **Generalization to Unseen Images:** Performance on the
        test dataset may not reflect performance on arbitrary
        internet images or real-world photographs.
        - **Confidence Interpretation:** Softmax confidence is not
        a guarantee of correctness. The model may produce highly
        confident incorrect predictions.
        - **Grad-CAM Limitations:** Heatmaps highlight influential
        image regions but do not prove that the model focuses on
        the correct features or explain its reasoning completely.
        - **No Unknown-Class Detection:** Images unrelated to the
        ten supported categories may still be assigned one of
        those categories.
        """
    )
