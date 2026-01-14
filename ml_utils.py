import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
import matplotlib.pyplot as plt

import torch.nn as nn
from transformers import ViTModel
from torchvision import models



# Configuration
st.set_page_config(page_title="Bone Age Predictor", page_icon="🦴", layout="wide")

# Age group information
AGE_GROUPS = {
    0: (0, 60, "0-5 years"),
    1: (60, 120, "5-10 years"),
    2: (120, 180, "10-15 years"),
    3: (180, 240, "15-20 years")
}

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 2.5em !important;
        color: #2E86C1;
        text-align: center;
        padding: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: #00000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .highlight {
        color: #28B463;
        font-weight: bold;
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 60%;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classification model
def load_classification_model():
    model = BoneAgeModel()  # Initialize the CNN-based classification model
    checkpoint = torch.load('Models/best_model.pth', map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])  # Load the state dictionary properly
    else:
        model.load_state_dict(checkpoint)  # Directly load if it's just a state_dict

    model.to(device)
    model.eval()
    return model


# Load regression model for specific age group
def load_regression_model(age_group):
    model_path = Path(f'Models/saved_models/best_models/age_group_{age_group}/best_model.pth')
    model = ViTBoneAgeModel(age_group).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Image preprocessing functions
def preprocess_for_classification(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

def preprocess_for_vit(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main app
def main():
    st.markdown('<div class="header">🦴 Pediatric Bone Age Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Patient Information")
        uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
        gender = st.radio("Gender", ["Male", "Female"], index=0)
        gender_tensor = torch.tensor([1 if gender == "Male" else 0], dtype=torch.float32).to(device)

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            with col1:
                st.markdown("### Uploaded X-ray Image")
                st.image(image, caption="Uploaded X-ray", use_column_width=True)

            # Classification preprocessing
            classification_image = preprocess_for_classification(image).to(device)
            
            # Load classification model
            classifier = load_classification_model()
            
            # Perform classification
            with torch.no_grad():
                class_output = classifier(classification_image, gender_tensor.unsqueeze(0))
                age_group = torch.argmax(class_output).item()

            # Regression preprocessing
            vit_image = Image.open(uploaded_file).convert('RGB')  # ViT expects color
            regression_image = preprocess_for_vit(vit_image).to(device)
            
            # Load regression model
            regressor = load_regression_model(age_group)
            
            # Perform regression
            with torch.no_grad():
                bone_age = regressor(regression_image, gender_tensor.unsqueeze(0)).item()

            # Display results
            with col2:
                st.markdown("### Prediction Results")
                
                age_group_info = AGE_GROUPS[age_group]
                months = round(bone_age, 1)
                years = round(months / 12, 1)
                
                with st.container():
                    st.markdown(f'<div class="result-box">'
                                f'<h4>📋 Prediction Summary</h4>'
                                f'<p>🔍 Detected Age Group: <span class="highlight">{age_group_info[2]}</span></p>'
                                f'<p>👶🏻➡️🧑🏻 Age Range: {age_group_info[0]}-{age_group_info[1]} months</p>'
                                f'<p>🦴 Predicted Bone Age: <span class="highlight">{months} months</span> ({years} years)</p>'
                                f'</div>', unsafe_allow_html=True)
                
                # Display age group visualization
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(0, age_group_info[1]-age_group_info[0], left=age_group_info[0], 
                       color='#2E86C1', height=0.5)
                ax.scatter(months, 0, color='#FF5733', s=200, zorder=5, 
                          label='Predicted Age')
                ax.set_yticks([])
                ax.set_xlabel('Age (months)')
                ax.set_xlim(0, 240)
                ax.legend(loc='upper center')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        # Show welcome message and instructions
        with col1:
            st.markdown("""
                ## Welcome to the Bone Age Predictor!
                
                **How to use:**
                1. Upload a pediatric hand X-ray image (JPEG/PNG)
                2. Select patient's gender
                3. View the predicted bone age and analysis
                
                **Features:**
                - 🧠 Deep learning-powered predictions
                - 📊 Visual age group representation
                - ⚕️ Clinical-grade accuracy
                """)
        
        with col2:
            st.markdown("### Sample X-ray Image")
            st.image("1386.png", caption="Sample Pediatric Hand X-ray", 
                    use_column_width=True)


class BoneAgeModel(nn.Module):
    def __init__(self):
        super(BoneAgeModel, self).__init__()
        
        # Pretrained DenseNet121 with modified input layer
        self.base_model = models.densenet121(pretrained=True)
        self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Fine-tuning more layers
        for param in self.base_model.features[:8].parameters():
            param.requires_grad = True
            
        # Feature extraction
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        
        # Image pathway
        self.image_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Gender pathway
        self.gender_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 4)
        )

    def forward(self, image, gender):
        image_features = self.base_model(image)
        image_features = self.image_fc(image_features)
        gender_features = self.gender_fc(gender)
        combined = torch.cat([image_features, gender_features], dim=1)
        return self.classifier(combined)

class ViTBoneAgeModel(nn.Module):
    def __init__(self, age_group, dropout_rate=0.3):
        super(ViTBoneAgeModel, self).__init__()
        self.age_group = age_group
        min_age, max_age, label = AGE_GROUPS[age_group]
        self.age_range = max_age - min_age
        
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Freeze early layers
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        num_unfrozen_layers = 2 + self.age_range // 40
        for param in self.vit.encoder.layer[-num_unfrozen_layers:].parameters():
            param.requires_grad = True

        # Gender embedding
        self.gender_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Combined regressor
        hidden_size = min(512, self.vit.config.hidden_size + 32)
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, gender):
        vit_output = self.vit(pixel_values=x).last_hidden_state[:, 0]
        gender_features = self.gender_embedding(gender)
        combined = torch.cat([vit_output, gender_features], dim=1)
        return self.regressor(combined)



if __name__ == "__main__":
    main()