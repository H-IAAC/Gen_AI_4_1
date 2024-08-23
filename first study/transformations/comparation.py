import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pyts.preprocessing import MinMaxScaler
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from scipy.signal import spectrogram
import pywt
from standartized_balanced import StandardizedBalancedDataset

# Ensure the CUDA device is set if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def noise_transform_vectorized(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

import os

class TimeSeriesVisualizer:
    def __init__(self, technique='recurrence', image_size=60, method='summation'):
        self.technique = technique
        self.image_size = image_size
        self.method = method
        self.transformer = self._get_transformer()
    
    def _get_transformer(self):
        if self.technique == 'recurrence':
            return RecurrencePlot(threshold='point', percentage=40)
        elif self.technique == 'gaf':
            return GramianAngularField(image_size=self.image_size, method=self.method)
        elif self.technique == 'mtf':
            return MarkovTransitionField(image_size=self.image_size)
        else:
            return None
    
    def transform_data(self, X):
        if self.technique == 'fft':
            return np.abs(np.fft.fft(X, axis=-1))  # Limit to image_size length
        elif self.technique == 'noise':
            return noise_transform_vectorized(X)
        elif self.technique == 'spectrogram':
            return np.array([spectrogram(x, nperseg=self.image_size)[2] for x in X])
        elif self.technique == 'wavelet':
            # Specify parameters for the cmor wavelet
            width = 2.0  # Bandwidth parameter
            center_frequency = 0.5  # Center frequency parameter
            return np.array([pywt.cwt(x, pywt.ContinuousWavelet(f'cmor{width}-{center_frequency}'), np.arange(1, 11))[0] for x in X])
        else:
            return self.transformer.fit_transform(X)
    
    def plot(self, X_train, y_train, num_samples_per_class=5, save_folder=None):
        # Get unique classes
        classes = np.unique(y_train)
        num_classes = len(classes)
        
        # Set up the figure for plotting
        fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(15, 2 * num_classes))
        fig.tight_layout()
        
        # Loop through each class
        for class_idx, cls in enumerate(classes):
            # Get indices of samples for the current class
            class_indices = np.where(y_train == cls)[0]
            
            # Select samples from the current class
            selected_indices = class_indices[:num_samples_per_class]
            
            for sample_idx, idx in enumerate(selected_indices):
                X = X_train[idx]
                
                # Apply the selected transformation
                X_transformed = self.transform_data(X.reshape(1, -1))  # Reshape X for transformer
                
                # Plot the transformed data
                ax = axes[class_idx, sample_idx]
                if self.technique in ['fft', 'spectrogram']:
                    if X_transformed[0].ndim == 1:
                        ax.plot(X_transformed[0])
                    else:
                        ax.imshow(X_transformed[0], cmap='viridis', origin='lower')
                elif self.technique == 'wavelet':
                    # For wavelets, ensure that the output is properly reshaped for plotting
                    X_wavelet = np.abs(X_transformed[0])
                    ax.imshow(X_wavelet, cmap='viridis', origin='lower')
                else:
                    ax.imshow(X_transformed[0], cmap='viridis', origin='lower')
                    
                ax.set_title(f'Class {cls}', fontsize=10)
                ax.axis('off')  # Hide axes for clarity
        
        # Adjust layout and save or show plot
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(os.path.join(save_folder, f'{dataset}_{self.technique}_visualization.png'))
        else:
            plt.show()
        plt.close()

def plot_accuracy_comparison(performance,dataset, save_path='plots/accuracy_comparison.png'):
    techniques = list(performance.keys())
    accuracies = list(performance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(techniques, accuracies, color='skyblue')
    plt.xlabel('Technique')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset} Comparison of Accuracy for Different Techniques')
    plt.ylim(0, 1)  # Assuming accuracy values are between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()


# Function to get data from the dataset
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val

# Ensure the CUDA device is set if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


datasets=["KuHar","MotionSense","RealWorld_thigh","RealWorld_waist","UCI","WISDM"]

for dataset in datasets:
    X_train, y_train, X_test, y_test, X_val, y_val = get_data(dataset, ['accel', 'gyro'], False)

    # Print the shape of the input data
    input_shape = X_train[0].shape
    print("Shape of X_train:", input_shape)

    # Normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize directories
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Initialize performance tracking
    performance = {}

    # Test different techniques
    techniques = ['fft', 'spectrogram', 'gaf', 'mtf']
    for technique in techniques:
        visualizer = TimeSeriesVisualizer(technique=technique, image_size=60)
        X_train_transformed = visualizer.transform_data(X_train)
        X_test_transformed = visualizer.transform_data(X_test)
        
        # Flatten images for SVM classifier
        X_train_flat = X_train_transformed.reshape(X_train_transformed.shape[0], -1)
        X_test_flat = X_test_transformed.reshape(X_test_transformed.shape[0], -1)
        
        # Train SVM classifier
        clf = SVC()
        clf.fit(X_train_flat, y_train)
        
        # Predict and evaluate accuracy
        y_pred = clf.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        performance[technique] = accuracy
        print(f'{dataset} Accuracy with {technique}: {accuracy:.2f}')
        
        # Save plot images for individual technique
        visualizer.plot(X_train, y_train, num_samples_per_class=5, save_folder=plot_dir)

    clf = SVC()
    X_train.reshape(X_train.shape[0], -1)
    X_test.reshape(X_test.shape[0], -1)

    clf.fit(X_train, y_train)
    technique='None'
        # Predict and evaluate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    performance[technique] = accuracy
    print(f'{dataset} Accuracy with {technique}: {accuracy:.2f}')

    # Save the accuracy comparison plot
    plot_accuracy_comparison(performance,dataset, save_path=os.path.join(plot_dir, f'{dataset}_accuracy_comparison.png'))

    # Print comparison results
    print("\nClassification Performance Comparison:")
    for technique, accuracy in performance.items():
        print(f'{technique}: {accuracy:.2f}')
