# Hair Classification System

A deep learning system for classifying **hair attributes** including:
- **Hair Type**: straight, wavy, curly
- **Hair Color**: [list your color categories]

This project uses state-of-the-art vision transformer and convolutional neural network architectures with transfer learning to achieve high accuracy on segmented hair images. The system features an **agnostic category recognition approach**, allowing flexible classification of different hair attributes using the same underlying architecture.

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
- [Citations](#citations)
- [License](#license)

## ✨ Features

- **Multi-Attribute Classification**: 
  - Hair type classification (straight, wavy, curly)
  - Hair color classification (blonde, brown, black, red, etc.)
  - Extensible to other hair attributes
- **Agnostic Category Recognition**: Unified architecture adaptable to different classification tasks
- **Modern Architecture Support**: ConvNeXt models
- **Transfer Learning**: Leverages ImageNet-1k and ImageNet-22k pre-trained weights
- **Advanced Training Techniques**: 
  - Class imbalance handling (weighted loss, balanced sampling)
  - MixUp and CutMix augmentation
  - Test-Time Augmentation (TTA)
- **Modular Codebase**: Easy to extend with new hair attributes
- **Production Ready**: Simple inference API with separate classifiers for each attribute

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Step 1: Clone repo  CUDA Support (GPU Users)

```bash
# Clone the repository
git clone https://github.com/LZ-sudo/hair_classification_model.git
cd hair_classification_model
```

### Step 2: Virtual env and CUDA Support (GPU Users)

```bash
# Create python virtual environment
python -m venv .venv

# Activate python virtual environment
.venv/Scripts/activate
```

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

### Step 3: Install Dependencies
```bash
# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation
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

Organize your segmented hair images in the following structure (e.g hair color):
```
input_pictures_color_processed/
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

Hair color data
```bash
python prepare_data.py --input_dir input_pictures_color_processed --output_dir data_hair_color
```
Hair type data
```bash
python prepare_data.py --input_dir input_pictures_type_processed --output_dir data_hair_type
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
dropout: 0.2
learning_rate: 0.00005
epochs: 80
# ... see config.yaml for full options
```

### Training

The system uses an **agnostic category recognition approach**, meaning you can train the same architecture on different classification tasks by simply:

1. Organizing your data by category
2. Updating the configuration
3. Running the training script

**Example: Training for Hair Type**
```yaml
# config.yaml
data_dir: './data_hair_type'              # Training data location
checkpoint_dir: './checkpoints_hair_type'  # Where to save model weights
input_dir: './input_pictures_hair_type'    # Raw images (for prepare_data.py)
# ... other settings
```

```bash
python train_model.py
```

**Example: Training for Hair Color**
```yaml
# config.yaml
data_dir: './data_hair_color'
checkpoint_dir: './checkpoints_hair_color'
input_dir: './input_pictures_hair_color'
# ... other settings
```

```bash
python train_model.py
```

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

For convnext_tiny/convnext_tiny_in22k models:

```bash
python classify_image_convnext.py --folder path/to/images/ # image batch

python classify_image_convnext.py --image image.jpg # single image
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
├── data_hair_color/                  # Generated by prepare_data.py, for data on hair colors
│   ├── train/
│   ├── val/
│   └── test/
│
├── data_hair_type/                   # Generated by prepare_data.py, for data on hair types
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
├── checkpoints_hair_color/           # Saved model weights for hair color classification training
│   └── best_model.pth
├── checkpoints_hair_type/            # Saved model weights for hair type classification training
│   └── best_model.pth
│
├── prepare_data.py                         # Split data into train/val/test
├── train_model.py                          # Main training script
├── test_model.py                           # Evaluation script
└── classify_image_convnext_hair_color.py   # Inference script for hair color classification (convnext_tiny/convnext_tiny_in22k)
└── classify_image_convnext_hair_type.py    # Inference script for hair type classification (convnext_tiny/convnext_tiny_in22k)
└── classify_image_convnext.py              # Inference script for both hair type and hair color (convnext_tiny/convnext_tiny_in22k)
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