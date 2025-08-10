# FlavorFusion Ingredient Classifier

FlavorFusion is a deep learning-powered web app for classifying food ingredients from images. Upload a photo of a fruit, vegetable, or spice, and the app predicts the ingredient and shows probability scores for all classes.

## Live Demo

Try FlavorFusion online: [https://flavourfusionmodel.streamlit.app/](https://flavourfusionmodel.streamlit.app/)

## Features

- Classifies images into 40+ ingredient categories
- Shows top 5 predictions with probabilities
- Visualizes probability distribution across all classes
- Easy-to-use Streamlit web interface

## Getting Started

### Prerequisites

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/flavorfusion.git
    cd flavorfusion
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download or place the trained model file `model_improved.pth` in the project root.

### Running the App

Start the Streamlit server:
```sh
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

## Usage

- Click "Upload an image..." and select a food ingredient photo (JPG/PNG).
- View the top 5 predictions and probability chart.

## Dataset

- Training, validation, and test images are in the [dataset/](dataset/) folder.
- Example images for testing are in [good_examples/](good_examples/).

## Model

- The model architecture is defined in [`ComplexNet`](app.py).
- Trained weights are loaded from `model_improved.pth`.
- Model training and evaluation code is in [ComplexNet.ipynb](ComplexNet.ipynb).

## File Structure

```
app.py                  # Main Streamlit app
requirements.txt        # Python dependencies
model_improved.pth      # Trained model weights
dataset/                # Image dataset (train/val/test)
good_examples/          # Example images for demo
ComplexNet.ipynb        # Model training notebook
CNN-Model Description.docx # Model documentation
```

## License

MIT License

## Acknowledgements

- PyTorch, Streamlit, torchvision
- Ingredient images from various sources

---

For questions or contributions, open an issue or pull