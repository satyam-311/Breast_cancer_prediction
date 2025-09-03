# %%
# 1. Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# 2. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target  # X = features, y = labels (0 = malignant, 1 = benign)

# %%
# 3. Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)  # scale features for better convergence


# %%
# convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # reshape to (n,1)


# %%
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# 4. Define Neural Network
class BreastCancerNN(nn.Module):
    def __init__(self, input_size):
        super(BreastCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)   # first hidden layer
        self.fc2 = nn.Linear(32, 16)           # second hidden layer
        self.fc3 = nn.Linear(16, 1)            # output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # probability of benign (1)
        return x


# %%
# initialize model
model = BreastCancerNN(X.shape[1])


# %%
# initialize model
model = BreastCancerNN(X.shape[1])

# %%
# 5. Define loss & optimizer
criterion = nn.BCELoss()  # binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# 6. Training loop
epochs = 50
for epoch in range(epochs):
    # forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
 # Save the trained model
torch.save(model.state_dict(), "breast_cancer_model.pth")



# %%
# 7. Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = (y_pred > 0.5).float()
    acc = (y_pred_classes.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"Accuracy: {acc*100:.2f}%")

# %%
import streamlit as st

# after training and evaluation
st.title("🩺 Breast Cancer Prediction using Neural Network")
st.write(f"Final Accuracy: {acc*100:.2f}%")

# simple user input demo
st.subheader("Enter Patient Data for Prediction")

user_inputs = []
for i, feature in enumerate(data.feature_names):
    val = st.number_input(f"{feature}", value=float(X[:, i].mean()), format="%.4f")
    user_inputs.append(val)

if st.button("Predict Cancer Status"):
    # scale inputs
    input_scaled = scaler.transform([user_inputs])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_tensor).item()

    if pred > 0.5:
        st.success("Prediction: **Benign (Safe)** ✅")
    else:
        st.error("Prediction: **Malignant (Cancerous)** ⚠️")

# %%




