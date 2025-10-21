# Hair Type Classification

A deep learning system for classifying hair types into three categories: **straight**, **wavy**, and **curly**. This project uses state-of-the-art vision transformer and convolutional neural network architectures with transfer learning to achieve high accuracy on segmented hair images.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Pre-trained Models](#pre-trained-models)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citations](#citations)
- [License](#license)

## ✨ Features

- **Multiple Architecture Support**: DeiT, DeiT-III, ConvNeXt models
- **Transfer Learning**: Leverages ImageNet-1k and ImageNet-22k pre-trained weights
- **Advanced Training Techniques**: 
  - Class imbalance handling (weighted loss, balanced sampling)
  - MixUp and CutMix augmentation
  - Test-Time Augmentation (TTA)
- **Modular Codebase**: Easy to extend and experiment with
- **Production Ready**: Simple inference API for deployment

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Step 1: CUDA Support (GPU Users)

**IMPORTANT**: If you have a CUDA-enabled GPU, install PyTorch with CUDA support **before** installing other dependencies.

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

To check your CUDA version:
```bash
nvidia-smi
```

### Step 2: Install Dependencies
```bash
# Clone the repository
git clone https://github.com/LZ-sudo/hair_classification_model.git
cd hair_classification_model

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 📊 Dataset

This project uses the **Hair Type Dataset** from Kaggle, which contains images of different hair types.

**Dataset Source**: [Hair Type Dataset on Kaggle](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset)

**Usage in this project**: 
- Only the **curly**, **straight**, and **wavy** subsets are used
- Images are split into training (70%), validation (15%), and testing (15%) sets
- All images are pre-processed using hair segmentation to isolate hair regions from backgrounds

**Citation**:
```
Kavya Sree B. (2023). Hair Type Dataset. Kaggle. 
https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset
```

**Note**: Please ensure you comply with the dataset's license terms when using this project.

## 📈 Model Performance

Performance comparison of different pre-trained models on the test set (with Test-Time Augmentation):

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| DeiT-Base | 86M | **92.42%** |
| DeiT3-Base | 86M | **91.41%** |
| ConvNeXt-Tiny | 28M | **93.43%** |
| ConvNeXt-Tiny (IN-22k) | 28M | **94.44%** |

**Key Findings**:
- ConvNeXt-Tiny achieves the best performance with fewer parameters
- ImageNet-22k pre-training (IN-22k) provides richer features for fine-grained classification
- Test-Time Augmentation typically adds 0.3-0.7% improvement

## 🤖 Pre-trained Models

This project leverages several state-of-the-art pre-trained models:

### 1. DeiT (Data-efficient Image Transformers)

**Paper**: "Training data-efficient image transformers & distillation through attention"  
**Authors**: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou  
**Conference**: ICML 2021  
```bibtex
@inproceedings{touvron2021training,
  title={Training data-efficient image transformers \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  pages={10347--10357},
  year={2021},
  organization={PMLR}
}
```

### 2. DeiT III

**Paper**: "DeiT III: Revenge of the ViT"  
**Authors**: Hugo Touvron, Matthieu Cord, Hervé Jégou  
**Conference**: ECCV 2022  
```bibtex
@inproceedings{touvron2022deit,
  title={Deit iii: Revenge of the vit},
  author={Touvron, Hugo and Cord, Matthieu and J{\'e}gou, Herv{\'e}},
  booktitle={European Conference on Computer Vision},
  pages={516--533},
  year={2022},
  organization={Springer}
}
```

### 3. ConvNeXt

**Paper**: "A ConvNet for the 2020s"  
**Authors**: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie  
**Conference**: CVPR 2022  
```bibtex
@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11976--11986},
  year={2022}
}
```

**Pre-training Datasets**:
- **ImageNet-1k**: 1.28M images, 1,000 classes
- **ImageNet-22k**: 14.2M images, 21,841 classes (provides richer features for transfer learning)

All models are loaded via the [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) library.

## 🚀 Usage

### Data Preparation

Organize your segmented hair images in the following structure:
```
input_pictures_processed/
├── straight/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── wavy/
│   ├── img101.jpg
│   ├── img102.jpg
│   └── ...
└── curly/
    ├── img201.jpg
    ├── img202.jpg
    └── ...
```

Then split the data into train/validation/test sets:
```bash
python prepare_data.py
```

This will create a `data/` directory with the following structure:
```
data/
├── train/
│   ├── straight/
│   ├── wavy/
│   └── curly/
├── val/
│   ├── straight/
│   ├── wavy/
│   └── curly/
└── test/
    ├── straight/
    ├── wavy/
    └── curly/
```

### Training

1. **Configure hyperparameters** in `config.yaml`:
```yaml
# Example for ConvNeXt-Tiny with ImageNet-22k
model_name: "convnext_tiny_in22k"
num_classes: 3
dropout: 0.2
learning_rate: 0.00005
epochs: 80
# ... see config.yaml for full options
```

2. **Train the model**:
```bash
python train_model.py
```

Training progress will be displayed in the console, and the best model will be saved to the `checkpoints/` directory.

### Testing

Evaluate the trained model on the test set:
```bash
python test_model.py
```

This will output:
- Classification report (precision, recall, F1-score per class)
- Confusion matrix
- Overall test accuracy

**With Test-Time Augmentation (recommended for best accuracy)**:
```yaml
# In config.yaml
use_tta: true
tta_augmentations: 5
```

### Inference


**Inference**:

For deit_base/deit3_base models:

```bash
python classify_image_deit_base.py --folder path/to/images/ --checkpoint checkpoints/deit3base_best_model.pth # image batch

python classify_image_deit_base.py --image image.jpg --checkpoint checkpoints/deit3base_best_model.pth # single image
```

For convnext_tiny/convnext_tiny_in22k models:

```bash
python classify_image_convnext_tiny.py --folder path/to/images/ --checkpoint checkpoints/convnext_tiny_best_model.pth # image batch

python classify_image_convnext_tiny.py --image image.jpg --checkpoint checkpoints/convnext_tiny_best_model.pth # single image
```

## 📁 Project Structure
```
hair_classification_model/
│
├── README.md
├── requirements.txt
├── config.yaml                       # Configuration file
├── .gitignore
│
├── data/                             # Generated by prepare_data.py
│   ├── train/
│   ├── val/
│   └── test/
│
├── input_pictures_processed/         # Your raw segmented images (place here)
│   ├── straight/
│   ├── wavy/
│   └── curly/
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── model.py                      # Model definitions
│   ├── dataset.py                    # Dataset and data loading
│   ├── train.py                      # Training utilities
│   └── predict.py                    # Inference utilities
│
├── checkpoints/                      # Saved model weights
│   └── best_model.pth
│
├── prepare_data.py                   # Split data into train/val/test
├── train_model.py                    # Main training script
├── test_model.py                     # Evaluation script
└── classify_image_convnext_tiny.py   # Inference script (convnext_tiny/convnext_tiny_in22k)
└── classify_image_deit_base.py       # Inference script (deit_base/deit3_base)
```

## ⚙️ Configuration

Key parameters in `config.yaml`:

### Model Settings
```yaml
model_name: "convnext_tiny_in22k"  # Model architecture
num_classes: 3                      # Number of hair types
dropout: 0.2                        # Dropout rate
```

### Training Settings
```yaml
epochs: 80                          # Number of training epochs
learning_rate: 0.00005              # Learning rate
weight_decay: 0.01                  # Weight decay for regularization
batch_size: 24                      # Batch size
freeze_epochs: 10                   # Epochs to freeze backbone
```

### Data Augmentation
```yaml
use_mixup: true                     # Enable MixUp augmentation
mixup_alpha: 0.2                    # MixUp alpha parameter
use_cutmix: true                    # Enable CutMix augmentation
cutmix_alpha: 1.0                   # CutMix alpha parameter
```

### Class Imbalance Handling
```yaml
use_class_weights: true             # Use weighted loss
use_balanced_sampling: false        # Alternative: balanced sampling
```

### Test-Time Augmentation
```yaml
use_tta: true                       # Enable TTA during testing
tta_augmentations: 5                # Number of TTA transforms
```

## 📚 Citations

If you use this code or models in your research, please cite the relevant papers:

**For DeiT models**:
```bibtex
@inproceedings{touvron2021training,
  title={Training data-efficient image transformers \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

**For ConvNeXt models**:
```bibtex
@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

**For the dataset**:
```
Kavya Sree B. (2023). Hair Type Dataset. Kaggle. 
https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset
```

## 📄 License

[Add your license here - MIT, Apache 2.0, etc.]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- **timm library**: For providing easy access to pre-trained models
- **PyTorch team**: For the deep learning framework
- **Kaggle & Kavya Sree B**: For the hair type dataset
- **Research communities**: For developing DeiT and ConvNeXt architectures

---

**Note**: This project is designed for segmented hair images (background removed). For best results, ensure input images have undergone hair segmentation preprocessing.