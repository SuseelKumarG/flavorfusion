🧠 AI for Ingredient Classification
Flavor Fusion utilizes a Vision Transformer (ViT) model to classify food ingredients — specifically spices — directly from images. This forms the backbone of the system's ability to understand and reason about ingredients in a recipe context.

🗂️ What the Model Does
Classifies images of individual ingredients (e.g., turmeric, spinach, cardamom) into 52 distinct spice and ingredient categories

Helps power the broader flavor fusion logic by accurately identifying the ingredient type from a photo

🏗️ Architecture & Workflow
Base Model: google/vit-base-patch16-224 from Hugging Face Transformers

Image Processing: ViTImageProcessor used for resizing and normalizing input images

Transfer Learning: Final classification head is replaced with a custom linear layer for 52 classes

Training Setup:

Optimizer: AdamW

Scheduler: Linear Warmup + Decay

Loss: CrossEntropy

Dropout Regularization: 0.4

Input Transformations: Resize → CenterCrop → ToTensor

🏋️ Dataset & Preprocessing
Custom Image Dataset organized in class-specific folders (ImageFolder format)

Dataset mounted from Google Drive and loaded with PyTorch Dataloaders

Train-test split: 80/20

Images are preprocessed using both torchvision.transforms and ViTImageProcessor

🔍 Inference Pipeline
Given a test image (e.g., Spinach_test.jpeg), the model:

Applies the same preprocessing as training

Performs a forward pass through the fine-tuned ViT model

Outputs the top predicted class from class_names (mapped from dataset)

💾 Model Persistence
Model state is saved as: /content/drive/MyDrive/pr dataset.pt

Can be reloaded for inference without retraining

