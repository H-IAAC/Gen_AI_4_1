import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import phate
import trimap
import pacmap
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from standartized_balanced import StandardizedBalancedDataset

# Function to load data using StandardizedBalancedDataset
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val

dataset_n="MotionSense"
X_train, y_train, X_test, y_test, X_val, y_val = get_data(dataset_n, ['accel', 'gyro'], False)
input_shape = X_train[0].shape
print(f"Input shape: {input_shape}")

def apply_fft(X):
    # Apply FFT to each sample and return the magnitude
    return np.abs(np.fft.fft(X, axis=-1))

X_train = apply_fft(X_train)
X_test = apply_fft(X_test)

# Normalize the data
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Reshape if needed
X_scaled_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))  # Reshape if needed

# Create 'plots' directory if it does not exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Define PacmapEmbedding class
class PacmapEmbedding:
    def __init__(self, n_components=2, n_neighbors=10):
        self.embedding_projector = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
        self.fitted = False

    def fit(self, data):
        emb = self.embedding_projector.fit_transform(data)
        self.fitted = True
        self.data_or = data
        return emb

    def transform(self, data):
        if not self.fitted:
            raise ValueError("You need to call the fit method before transform")
        return self.embedding_projector.transform(data, self.data_or)

    def get_embedding(self):
        if not self.fitted:
            raise ValueError("You need to call the fit method before get_embedding")
        return self.embedding_projector.transform(self.data_or)

# Define Autoencoder model using PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Helper function to perform classification and evaluation
def evaluate_classifier(X_train, y_train, X_test, y_test, title):
    clf = SVC(kernel='linear')  # Linear kernel for simplicity
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[title] = accuracy
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report for {title}:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title(f"Confusion Matrix for {title}")
    plt.savefig(f"plots/confusion_matrix_{title.lower().replace(' ', '_')}.jpg")
    plt.show()

# Train and transform data using Autoencoder
def train_autoencoder(X_train, X_test, encoding_dim):
    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Convert data to DataLoader for batching
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Training loop
    epochs = 50
    autoencoder.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Apply the encoder to reduce dimensionality
    autoencoder.eval()
    with torch.no_grad():
        X_autoencoder_train = autoencoder.get_encoded(X_train_tensor).numpy()
        X_autoencoder_test = autoencoder.get_encoded(X_test_tensor).numpy()
    
    return X_autoencoder_train, X_autoencoder_test

# Dictionary to store accuracy scores
accuracy_scores = {}

# Dimensions to test
dimensions = [2, 6, 10, 100,300]

# Apply dimensionality reduction techniques for different dimensions
for dim in dimensions:
    print(f"Testing with {dim} dimensions")

    # PHATE
    phate_operator = phate.PHATE(n_components=dim)
    X_phate_train = phate_operator.fit_transform(X_scaled_train)
    X_phate_test = phate_operator.transform(X_scaled_test)
    evaluate_classifier(X_phate_train, y_train, X_phate_test, y_test, f"PHATE_{dim}")

    # TRIMAP
    X_trimap_train = trimap.TRIMAP(n_dims=dim).fit_transform(X_scaled_train)
    X_trimap_test = trimap.TRIMAP(n_dims=dim).fit_transform(X_scaled_test)
    evaluate_classifier(X_trimap_train, y_train, X_trimap_test, y_test, f"TRIMAP_{dim}")

    # PacMAP
    pacmap_embedding = PacmapEmbedding(n_components=dim, n_neighbors=10)
    X_pacmap_train = pacmap_embedding.fit(X_scaled_train)
    X_pacmap_test = pacmap_embedding.transform(X_scaled_test)
    evaluate_classifier(X_pacmap_train, y_train, X_pacmap_test, y_test, f"PacMAP_{dim}")

    # UMAP
    umap_operator = UMAP(n_components=dim)
    X_umap_train = umap_operator.fit_transform(X_scaled_train)
    X_umap_test = umap_operator.transform(X_scaled_test)
    evaluate_classifier(X_umap_train, y_train, X_umap_test, y_test, f"UMAP_{dim}")

    # TruncatedSVD
    svd_operator = TruncatedSVD(n_components=dim)
    X_svd_train = svd_operator.fit_transform(X_scaled_train)
    X_svd_test = svd_operator.transform(X_scaled_test)
    evaluate_classifier(X_svd_train, y_train, X_svd_test, y_test, f"TruncatedSVD_{dim}")

    # Word2Vec Embedding
    X_train_sequences = [list(map(str, x.flatten())) for x in X_train]
    X_test_sequences = [list(map(str, x.flatten())) for x in X_test]

    # Training Word2Vec model
    # w2v_model = Word2Vec(sentences=X_train_sequences, vector_size=dim, window=5, min_count=1, sg=1)

    # Generating embeddings for each sequence
    # X_w2v_train = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_train_sequences])
    # X_w2v_test = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_test_sequences])
    # evaluate_classifier(X_w2v_train, y_train, X_w2v_test, y_test, f"Word2Vec_{dim}")

    # Autoencoder
    X_autoencoder_train, X_autoencoder_test = train_autoencoder(X_scaled_train, X_scaled_test, encoding_dim=dim)
    evaluate_classifier(X_autoencoder_train, y_train, X_autoencoder_test, y_test, f"Autoencoder_{dim}")

# Plot comparative accuracy
accuracy_df = pd.DataFrame(accuracy_scores.items(), columns=['Technique', 'Accuracy'])
fig = px.bar(accuracy_df, x='Technique', y='Accuracy', title=f"{dataset_n} Accuracy Scores for Different Techniques")
fig.write_html(f'plots/{dataset_n}_accuracy_comparison.html')
fig.write_image(f'plots/{dataset_n}_accuracy_comparison.png')
