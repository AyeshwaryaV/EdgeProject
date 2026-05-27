# train_nn_q1.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("🧠 INITIALIZING DEEP NON-LINEAR NEURAL SEPARATION NETWORK")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load Data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X_raw = df[features].values
y_raw = df['Engine Condition'].values

# Advanced Mapping: Quantile Transformation spreads overlapping distributions smoothly
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled = scaler.fit_transform(X_raw)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Convert arrays to PyTorch Tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Define High-Dimensional Separation MLP
class DiagnosticMLP(nn.Module):
    def __init__(self, input_dim):
        super(DiagnosticMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

model = DiagnosticMLP(len(features))
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

# 3. Network Training Loop
print("🏋️ Training neural mapping layers across coordinate planes...")
model.train()
for epoch in range(35):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x).squeeze()
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

# 4. Run System Evaluation
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_probs_tensor = model(X_test_tensor).squeeze()
    y_probs = y_probs_tensor.numpy()
    y_pred = (y_probs > 0.5).astype(int)

# Compute Core Verification Scores
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 NEW NEURAL NETWORK SEPARATION RESULTS:")
print(f" ├─ Target Accuracy:  {acc:.2%}")
print(f" ├─ System Precision: {prec:.2%}")
print(f" ├─ System Recall:    {rec:.2%}")
print(f" └─ Journal F1-Score: {f1:.4f}")

print("\n📋 NEURAL MAPPING DISTRIBUTION MATRIX:")
print(f" True Normals Caught (TN): {cm[0,0]} | False Alarms (FP): {cm[0,1]}")
print(f" Missed Faults (FN):         {cm[1,0]} | Actual Faults Caught (TP): {cm[1,1]}")

# Save crisp visual for your draft chapter
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: Neural Network Confusion Matrix')
plt.ylabel('True Class Label')
plt.xlabel('Edge System Prediction')
plt.tight_layout()
plt.savefig('plot_confusion_matrix_nn.png', dpi=300)

# Save the necessary ROC Curve Plot
plt.figure(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC Curve (Area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4.5: Neural Network ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plot_roc_curve_nn.png', dpi=300)

print("\n🖼️ Verification charts generated successfully:")
print(" ├─ plot_confusion_matrix_nn.png")
print(" └─ plot_roc_curve_nn.png")
print("="*60)