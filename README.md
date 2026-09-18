# ☀️ CNN-Based Solar Panel Fault Detection

A deep learning project that classifies photos of solar panels into **six condition categories** using a **custom Convolutional Neural Network built from scratch in PyTorch** (no pretrained models, no transfer learning).

Dust, bird droppings, snow, and physical or electrical damage all reduce the energy output of a solar panel. Detecting them early from a simple photo makes maintenance faster and cheaper than manual inspection. This project covers the full pipeline: dataset preparation, image preprocessing, augmentation, model training, and evaluation.

---

## 🎯 Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | `Bird-drop` | Panel covered by bird droppings |
| 2 | `Clean` | Healthy, clean panel |
| 3 | `Dusty` | Dust or dirt accumulation |
| 4 | `Electrical-damage` | Burn marks or electrical faults |
| 5 | `Physical-Damage` | Cracks or broken glass |
| 6 | `Snow-Covered` | Panel covered by snow |

## 📊 Dataset

Source: **Solar Panel Images** dataset on Kaggle (`Faulty_solar_panel` folder). Please credit the original dataset author.

| Class | Original images | Train | Val | Test |
|-------|:---:|:---:|:---:|:---:|
| Bird-drop | 171 | 136 | 17 | 17 |
| Clean | 185 | 148 | 18 | 19 |
| Dusty | 190 | 152 | 19 | 19 |
| Electrical-damage | 90 | 71 | 8 | 10 |
| Physical-Damage | 66 | 52 | 6 | 8 |
| Snow-Covered | 105 | 84 | 10 | 11 |
| **Total** | **807** | **643** | **78** | **84** |

The split is **80 / 10 / 10**, done per class. Augmentation is applied **only to the training set** (after splitting), so no augmented copy of a training image can leak into validation or test.

## ⚙️ Pipeline

### Phase 1: Data preparation
1. **Verification:** count and visualise images per class.
2. **Preprocessing:** resize to **224×224**, apply **CLAHE** on the luminance channel (clip limit 2.0, 8×8 tiles) to improve contrast, then **Non-Local Means denoising**.
3. **Quality metrics:** PSNR and SSIM measure the effect of preprocessing (PSNR 19.6-25.2 dB, SSIM 0.83-0.88 across classes).
4. **Split:** stratified per class into train, validation and test.
5. **Augmentation:** random flip, rotation (±20°), zoom (0.8-1.2×) and brightness (0.7-1.3×), 1-2 operations per image. This balances the training set to **1,000 images per class (6,000 total)**.

### Phase 2: Custom CNN

```
Input (3×224×224)
 → [Conv3×3 → ReLU → BatchNorm → MaxPool] × 4   (32 → 64 → 128 → 256 filters)
 → AdaptiveAvgPool (7×7) → Flatten
 → Dropout(0.5) → Linear(12544→1024) → ReLU
 → Dropout(0.5) → Linear(1024→6)
```

About **13.2 million trainable parameters**.

| Setting | Value |
|---------|-------|
| Framework | PyTorch 2.6 (CUDA) |
| Optimizer | AdamW (lr = 1e-4, weight decay = 1e-4) |
| Loss | Cross-entropy |
| Batch size | 32 |
| LR scheduler | ReduceLROnPlateau (factor 0.2, patience 3) |
| Early stopping | Patience 5 on validation loss |
| Training time | ~97 seconds on a Kaggle GPU (stopped after 6 epochs) |

## 📈 Results

| Split | Accuracy | Macro F1 |
|-------|:---:|:---:|
| Train (augmented) | 86.28% | 0.863 |
| Validation | 69.23% | 0.680 |
| **Test** | **67.86%** | **0.674** |

Test inference time for all 84 images: **0.25 s**.

**Per-class test performance**

| Class | Precision | Recall | F1 | Support |
|-------|:---:|:---:|:---:|:---:|
| Bird-drop | 0.50 | 0.65 | 0.56 | 17 |
| Clean | 0.78 | 0.74 | 0.76 | 19 |
| Dusty | 0.64 | 0.74 | 0.68 | 19 |
| Electrical-damage | 0.78 | 0.70 | 0.74 | 10 |
| Physical-Damage | 0.60 | 0.38 | 0.46 | 8 |
| Snow-Covered | 1.00 | 0.73 | 0.84 | 11 |

Snow-covered panels are the easiest class to recognise. Physical damage and bird droppings are the hardest, probably because they are small, varied and visually similar to other classes, and because the dataset has few examples of them.

## ⚠️ Limitations

- **Small dataset:** only 807 original images, and the validation and test sets have just 78 and 84 images. A single image changes accuracy by more than 1%, so treat these numbers as indicative, not definitive.
- **Class imbalance:** Physical-Damage has only 66 originals. Augmentation balances the counts but cannot add genuinely new information.
- **Overfitting:** training accuracy is well above validation and test accuracy, and validation loss did not improve after the first epoch.
- Trained from scratch, so the model has no prior visual knowledge.

## 🚀 Future work

- Fine-tune pretrained models (ResNet, EfficientNet, MobileNet) and compare them with this baseline.
- Collect more real images, especially for physical and electrical damage.
- Use k-fold cross-validation for more reliable estimates on a small dataset.
- Add Grad-CAM visualisations to show which regions drive each prediction.
- Deploy as a web or mobile app, or run it on drone imagery for large solar farms.

## ▶️ How to run

**On Kaggle (recommended):** open the notebook, attach the *Solar Panel Images* dataset, enable a GPU, and use **Run All**.

**Locally:**
```bash
git clone https://github.com/<your-username>/solar-panel-fault-detection-cnn.git
cd solar-panel-fault-detection-cnn
pip install torch torchvision opencv-python scikit-image scikit-learn seaborn matplotlib tqdm pillow numpy pandas
```
Download the dataset, then change `dataset_path` in Phase 1 and the directory paths in the Phase 2 configuration cell to your local folders.

## 📁 Repository structure

```
├── solar_panel_fault_detection.ipynb   # Full pipeline: preprocessing → training → evaluation
├── README.md
└── (optional) models/best_custom_cnn.pth
```

## 🛠 Tech stack

Python · PyTorch · torchvision · OpenCV · scikit-image · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn

## 👤 Author

**Muhammad Hassan**: [Kaggle](https://www.kaggle.com/muhammadhassan82) · [GitHub](https://github.com/<your-username>)

## 📄 License

Released under the MIT License. The dataset belongs to its original author and is subject to its own license.
