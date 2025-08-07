import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class ComplexNet(nn.Module):
    def __init__(self, num_classes):
        super(ComplexNet, self).__init__()
        def cbr_block(in_f, out_f, kernel, stride=1):
            padding_val = (kernel - 1) // 2
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=kernel, stride=stride, padding=padding_val),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True)
            )

        self.conv1 = cbr_block(3, 16, 3)
        self.conv2 = cbr_block(16, 16, 5, 2)
        self.inc1_3 = cbr_block(16, 16, 3)
        self.inc1_5 = cbr_block(16, 16, 5)
        self.inc1_7 = cbr_block(16, 16, 7)
        self.inc1_11 = cbr_block(16, 16, 11)
        self.bn_cat1 = nn.BatchNorm2d(16 + 16 * 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.inc2_3 = cbr_block(self.bn_cat1.num_features, 32, 3, 2)
        self.inc2_5 = cbr_block(self.bn_cat1.num_features, 32, 5, 2)
        self.inc2_7 = cbr_block(self.bn_cat1.num_features, 32, 7, 2)
        self.inc2_11 = cbr_block(self.bn_cat1.num_features, 32, 11, 2)
        self.bn_cat2 = nn.BatchNorm2d(32 * 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.inc3_3 = cbr_block(self.bn_cat2.num_features, 32, 3)
        self.inc3_5 = cbr_block(self.bn_cat2.num_features, 32, 5)
        self.inc3_7 = cbr_block(self.bn_cat2.num_features, 32, 7)
        self.inc3_11 = cbr_block(self.bn_cat2.num_features, 32, 11)
        self.bn_cat3 = nn.BatchNorm2d(self.bn_cat2.num_features + 32 * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.inc4_3 = cbr_block(self.bn_cat3.num_features, 64, 3, 2)
        self.inc4_5 = cbr_block(self.bn_cat3.num_features, 64, 5, 2)
        self.inc4_7 = cbr_block(self.bn_cat3.num_features, 64, 7, 2)
        self.inc4_11 = cbr_block(self.bn_cat3.num_features, 64, 11, 2)
        self.skip_conv = cbr_block(self.bn_cat1.num_features, 64, 7, 4)
        self.bn_cat4 = nn.BatchNorm2d(64 * 4 + 64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.bn_cat4.num_features, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1_3 = self.inc1_3(x)
        x1_5 = self.inc1_5(x)
        x1_7 = self.inc1_7(x)
        x1_11 = self.inc1_11(x)
        cat1 = torch.cat([x, x1_3, x1_5, x1_7, x1_11], dim=1)
        bn_cat1 = self.bn_cat1(cat1)
        pool1 = self.pool1(bn_cat1)
        x2_3 = self.inc2_3(pool1)
        x2_5 = self.inc2_5(pool1)
        x2_7 = self.inc2_7(pool1)
        x2_11 = self.inc2_11(pool1)
        cat2 = torch.cat([x2_3, x2_5, x2_7, x2_11], dim=1)
        bn_cat2 = self.bn_cat2(cat2)
        pool2 = self.pool2(bn_cat2)
        x3_3 = self.inc3_3(pool2)
        x3_5 = self.inc3_5(pool2)
        x3_7 = self.inc3_7(pool2)
        x3_11 = self.inc3_11(pool2)
        cat3 = torch.cat([pool2, x3_3, x3_5, x3_7, x3_11], dim=1)
        bn_cat3 = self.bn_cat3(cat3)
        pool3 = self.pool3(bn_cat3)
        x4_3 = self.inc4_3(pool3)
        x4_5 = self.inc4_5(pool3)
        x4_7 = self.inc4_7(pool3)
        x4_11 = self.inc4_11(pool3)
        skip_out = self.skip_conv(bn_cat1)
        skip_out_resized = nn.functional.interpolate(skip_out, size=(x4_3.shape[2], x4_3.shape[3]), mode='bilinear', align_corners=False)
        cat4 = torch.cat([x4_3, x4_5, x4_7, x4_11, skip_out_resized], dim=1)
        bn_cat4 = self.bn_cat4(cat4)
        out = self.avg_pool(bn_cat4)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

class_names = [
    'apple', 'banana', 'beetroot', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'cinnamon', 'cloves', 'corn', 'corriander', 'corriander_seeds',
    'cucumber', 'cumin', 'eggplant', 'fennel_seeds', 'garlic', 'ginger', 'grapes',
    'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'nutmeg', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'saffron', 'soy beans', 'spinach', 'star_anise', 'sweetpotato', 'tamarind',
    'tomato', 'turmeric', 'turnip', 'watermelon'
]


@st.cache_resource
def load_model():
    model = ComplexNet(num_classes=len(class_names))
    model.load_state_dict(torch.load('model_improved.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🍉 Fruit & Veggie Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_prob, top5_indices = torch.topk(probabilities, 5)

    st.subheader("🔍 Top 5 Predictions:")
    for i in range(5):
        label = class_names[top5_indices[i].item()]
        prob = top5_prob[i].item() * 100
        st.write(f"**{label}**: {prob:.2f}%")
    st.subheader("📊 Probability Across All Classes")
    all_probs = probabilities.numpy()
    sorted_indices = all_probs.argsort()[::-1]
    sorted_probs = all_probs[sorted_indices]
    sorted_labels = [class_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 18))
    ax.barh(sorted_labels, sorted_probs, color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities")

    st.pyplot(fig)
