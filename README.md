# Image Classification – Fake vs Real Chameleons

## Project Summary
Fine-tuned EfficientNet-B0 to distinguish AI-generated chameleon images (“fake”) from real ones. Starting from a 26k-image dataset (11,170 fake / 14,863 real), I froze the feature extractor and compared two lightweight classification heads:

- **Model A (GAP)**: keeps EfficientNet’s global average pooling head for better robustness to varying input sizes.
- **Model B (Flatten)**: removes GAP, flattens the 7×7×1280 feature map, and adds a linear layer for raw capacity.

Although Model B achieves higher training accuracy (~97%), Model A generalizes better to arbitrary resolutions, making it the safer choice for real-world deployments.

## Dataset & Preprocessing
- Source: https://github.com/shilinyan99/AIDE/issues/7
- Split: 80/20 stratified random split → 20,826 training images / 5,207 validation images.
- Transforms: Resize to 224×224, convert to tensor, normalize with ImageNet mean/std.
- DataLoader: batch size 32, shuffle enabled for training loader.

## Training Configuration
- **Backbone**: `torchvision.models.efficientnet_b0` with pretrained weights (`weights='DEFAULT'`).
- **Optimization**: Adam, learning rate 1e-4, 10 epochs, cross-entropy loss.
- **Hardware**: CUDA-enabled runtime (Colab GPU).
- **Robustness Test**: fed 300×300 validation samples to both models to evaluate shape handling.

## Results & Findings
| Model | Head | Train Acc (Epoch 10) | Robustness Outcome | Notes |
|-------|------|----------------------|--------------------|-------|
| Model A | Global Average Pooling + Linear | 88.4% | ✅ Handles 300×300 inputs | Slightly lower accuracy but size-agnostic |
| Model B | Flattened 7×7×1280 + Linear | **96.6%** | ❌ Shape mismatch on 300×300 | High capacity yet brittle outside 224×224 |

- **Key insight**: Removing GAP inflates parameter count (over 12M weights in the classifier alone) and ties the model to a fixed spatial dimension. GAP-based head keeps the receptive field benefits of EfficientNet and accepts any resolution.
- **Class imbalance**: Balanced by augmentation-free sampling; both classes nearly balanced so no weighted loss required.
- **Inference demo**: Random validation samples show Model A’s predictions with >90% confidence while maintaining correct labels even when resized.

## Visualizations
### Training Accuracy Comparison
![Training Accuracy](training_accuracy.png)
The plot compares the two heads over 10 epochs. Model B converges faster but sacrifices robustness; Model A plateaus around 88% yet remains input-size invariant.


This project demonstrates transfer learning, experimentation with classifier heads, and robustness validation for image classification workloads.

