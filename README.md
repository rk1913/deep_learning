# 👕 FashionMNIST ANN Classifier using PyTorch

A simple Artificial Neural Network (ANN) built with **PyTorch** to classify clothing images from the FashionMNIST dataset.

This project demonstrates the complete deep learning workflow, including:

- Loading datasets
- Data preprocessing
- Building a neural network
- Training
- Evaluating accuracy
- Saving and loading the model
- Predicting on unseen images

---

## 📂 Project Structure

```
FashionMNIST_ANN/
│
├── data/                    # Dataset (downloaded automatically)
├── models/
│   └── fashion_mnist_ann.pth
│
├── fashion_mnist_ann.py     # Main training script
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

**Dataset:** FashionMNIST

- 60,000 Training Images
- 10,000 Testing Images
- 28 × 28 grayscale images
- 10 Clothing Categories

Classes:

| Label | Class |
|------:|----------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## 🧠 Model Architecture

```
Input Image
(1 × 28 × 28)

        │

Flatten (784)

        │

Linear(784 → 512)

        │

ReLU

        │

Linear(512 → 128)

        │

ReLU

        │

Linear(128 → 10)

        │

Output Logits
```

---

## ⚙️ Technologies Used

- Python
- PyTorch
- TorchVision

---

## 🚀 Features

- Automatic dataset download
- GPU / Apple Silicon (MPS) support
- Custom ANN architecture
- CrossEntropyLoss
- Adam Optimizer
- Training and Testing pipeline
- Model saving/loading
- Single image prediction

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/rk1913/deep_learning.git
```

Move into the project

```bash
cd deep_learning
```

Install dependencies

```bash
pip install torch torchvision
```

---

## ▶️ Run

```bash
python fashion_mnist_ann.py
```

---

## 💾 Saving Model

The trained model is saved as

```
fashion_mnist_ann.pth
```

Load it later using

```python
model.load_state_dict(torch.load("fashion_mnist_ann.pth"))
model.eval()
```

---

## 📈 Evaluation

The model is evaluated on the FashionMNIST test dataset.

Metrics:

- Test Accuracy
- Single Image Prediction

---

## 🖼️ Example Prediction

```
Actual      : Sneaker
Prediction  : Sneaker
```

---

## 📚 Concepts Practiced

- Dataset Loading
- DataLoader
- Batch Processing
- Tensor Operations
- Forward Propagation
- Backpropagation
- Loss Functions
- Optimizers
- Model Evaluation
- Model Persistence
- Device Management (CPU / CUDA / Apple MPS)

---

## 🎯 Learning Outcome

This project provides a complete introduction to image classification using a fully connected neural network (ANN). It serves as a strong foundation before moving to Convolutional Neural Networks (CNNs), where spatial feature extraction significantly improves image classification performance.

---

## 👨‍💻 Author

**Sampath Rajana**

GitHub: https://github.com/rk1913