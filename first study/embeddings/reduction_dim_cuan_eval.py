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
from standartized_balanced import StandardizedBalancedDataset
import matplotlib.pyplot as plt

# Function to load data using StandardizedBalancedDataset
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val


dataset="UCI"
X_train, y_train, X_test, y_test, X_val, y_val = get_data(dataset, ['accel', 'gyro'], False)
input_shape = X_train[0].shape
print(f"Input shape: {input_shape}")

def apply_fft(X):
    # Aplicar FFT em cada amostra e retornar o módulo da FFT (magnitude)
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
            raise ValueError("Você precisa chamar o método fit antes de chamar o método transform")
        return self.embedding_projector.transform(data, self.data_or)

    def get_embedding(self):
        if not self.fitted:
            raise ValueError("Você precisa chamar o método fit antes de chamar o método get_embedding")
        return self.embedding_projector.transform(self.data_or)

# Dictionary to store accuracy scores
accuracy_scores = {}

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

# Dimensions to test
dimensions = [2, 6, 10, 100]

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
    #X_w2v_train = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_train_sequences])
    #X_w2v_test = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_test_sequences])
    #evaluate_classifier(X_w2v_train, y_train, X_w2v_test, y_test, f"Word2Vec_{dim}")

# Plot comparative accuracy
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Technique', 'Accuracy'])
fig = px.bar(accuracy_df, x='Technique', y='Accuracy', title=f"{dataset} Comparative Accuracy of Dimensionality Reduction Techniques")
fig.write_html(f"plots/{dataset}comparative_accuracy.html")
fig.write_image("plots/{dataset}comparative_accuracy.jpg")
fig.show()
