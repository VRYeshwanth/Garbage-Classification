import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    """
    Reusable Grad-CAM utility for PyTorch classification models.

    Expected input:
        A normalized image tensor with shape (C, H, W).

    The target layer must belong to the supplied model.
    """

    def __init__(
        self,
        model,
        target_layers,
        class_names,
        device,
        mean=None,
        std=None
    ):
        self.model = model
        self.target_layers = target_layers
        self.class_names = class_names
        self.device = device

        self.mean = np.array(
            mean or [0.485, 0.456, 0.406],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            std or [0.229, 0.224, 0.225],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.model.eval()

    def denormalize(self, image_tensor):
        """Convert a normalized CHW tensor to an RGB image in [0, 1]."""

        image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()

        image = image * self.std + self.mean
        image = np.clip(image, 0, 1)

        return image.astype(np.float32)

    
    def explain(self, image_tensor, target_class=None, original_image=None):
        """
        Generate Grad-CAM for a normalized image tensor.

        image_tensor:
            Normalized tensor (C, H, W) used for model inference.

        target_class:
            Class index to explain. Defaults to the predicted class.

        original_image:
            Optional original-resolution RGB PIL image. When supplied,
            the overlay is generated at the original image dimensions.
        """

        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Get prediction and confidence.
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        if not 0 <= target_class < len(self.class_names):
            raise ValueError("Invalid target class index.")

        targets = [ClassifierOutputTarget(target_class)]

        # Grad-CAM runs on the resized model input.
        with GradCAM(
            model=self.model,
            target_layers=self.target_layers
        ) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

        # Prepare the image for visualization.
        if original_image is not None:
            rgb_image = np.asarray(
                original_image.convert("RGB"),
                dtype=np.float32
            ) / 255.0
        else:
            rgb_image = self.denormalize(image_tensor)

        # Resize the heatmap to the displayed image dimensions.
        height, width = rgb_image.shape[:2]

        heatmap_image = Image.fromarray(
            np.uint8(np.clip(grayscale_cam, 0, 1) * 255)
        )

        heatmap_image = heatmap_image.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR
        )

        grayscale_cam = (
            np.asarray(heatmap_image, dtype=np.float32) / 255.0
        )

        # Overlay and original now have matching dimensions.
        overlay = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6
        )

        return {
            "original": rgb_image,
            "heatmap": grayscale_cam,
            "overlay": overlay,
            "predicted_class": predicted_class,
            "predicted_name": self.class_names[predicted_class],
            "confidence": confidence,
            "target_class": target_class,
            "target_name": self.class_names[target_class],
        }

    def show(self, result, true_label=None, figsize=(15, 6)):
        """Display the original image, heatmap and overlay."""

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(result["original"])

        title = "Original Image"
        if true_label is not None:
            title += f"\nTrue: {self.class_names[true_label]}"

        axes[0].set_title(title)

        axes[1].imshow(
            result["heatmap"],
            cmap="jet",
            vmin=0,
            vmax=1
        )
        axes[1].set_title(
            f"Grad-CAM: {result['target_name']}"
        )

        axes[2].imshow(result["overlay"])
        axes[2].set_title(
            f"Predicted: {result['predicted_name']}\n"
            f"Confidence: {result['confidence']:.2%}"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
