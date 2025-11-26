#!/usr/bin/env python3
"""
Generate training accuracy comparison plot for EfficientNet experiments.
"""
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 11)
acc_gap = np.array([88.03, 88.57, 88.33, 88.54, 88.41, 88.60, 88.78, 87.98, 88.57, 88.43])
acc_flat = np.array([87.11, 92.78, 94.00, 95.23, 95.59, 96.03, 96.14, 96.77, 97.01, 96.58])

plt.figure(figsize=(8, 5))
plt.plot(epochs, acc_gap, marker='o', label='Model A: EfficientNet + GAP')
plt.plot(epochs, acc_flat, marker='s', label='Model B: EfficientNet + Flatten')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.ylim(85, 98)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=300)
print("Saved training_accuracy.png")

