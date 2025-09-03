# bcd.py
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import streamlit as st

# -----------------------------
# 1. Load dataset (for scaler + feature names)
# -----------------------------
data = load_breast_cancer()
X, y = data.data, data.target

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# 2. Define Neural Network
# -----------------------------
class BreastCancerNN(nn.Module):
    def __init__(self, input_size):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# -----------------------------
# 3. Load trained model
# -----------------------------
model = BreastCancerNN(X.shape[1])
model.load_state_dict(torch.load("breast_cancer_model.pth", map_location=torch.device("cpu")))
model.eval()

# -----------------------------
# 4. Streamlit App
# -----------------------------
st.title("🩺 Breast Cancer Prediction using Neural Network")
st.write("Enter patient diagnostic features to predict whether the tumor is **Benign** or **Malignant**.")

# input fields for all features
user_inputs = []
for i, feature in enumerate(data.feature_names):
    val = st.number_input(f"{feature}", value=float(X[:, i].mean()), format="%.4f")
    user_inputs.append(val)

# prediction button
if st.button("🔍 Predict Cancer Status"):
    # scale input
    input_scaled = scaler.transform([user_inputs])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # prediction
    with torch.no_grad():
        pred = model(input_tensor).item()

    if pred > 0.5:
        st.success("✅ Prediction: **Benign (Safe)**")
    else:
        st.error("⚠️ Prediction: **Malignant (Cancerous)**")

