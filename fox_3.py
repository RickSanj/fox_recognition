import os
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

def load_images_and_labels(base_dir, categories, shuffle_data=False):
    images = []
    labels = []  # 1 for 'fox', 0 for 'not_fox'
    for label, category in categories.items():
        dir_path = os.path.join(base_dir, category)
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    if shuffle_data:
        images, labels = shuffle(images, labels, random_state=42)

    return images, labels

def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray_img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.zeros((0, 128)))  # Append an empty array for images with no descriptors
    return descriptors_list

def train_kmeans(descriptors_list, n_clusters=500):
    all_descriptors = np.vstack([desc for desc in descriptors_list if desc.size > 0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

def create_histograms(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors.size > 0:
            labels = kmeans.predict(descriptors)
            hist = np.histogram(labels, bins=np.arange(kmeans.n_clusters+1))[0]
        else:
            hist = np.zeros(kmeans.n_clusters)
        histograms.append(hist)
    return np.array(histograms)

# Paths to your dataset
test_dir = './test'
train_dir = './train'
categories = {0: 'not_fox', 1: 'fox'}

# Load images and labels
train_images, train_labels = load_images_and_labels(train_dir, categories,  shuffle_data=True)
test_images, test_labels = load_images_and_labels(test_dir, categories, shuffle_data=True)

# Extract SIFT descriptors
train_descriptors = extract_sift_descriptors(train_images)
test_descriptors = extract_sift_descriptors(test_images)

# Train k-means to get visual words
kmeans = train_kmeans(train_descriptors)

# Create histograms of visual words
train_features = create_histograms(train_descriptors, kmeans)
test_features = create_histograms(test_descriptors, kmeans)

# Feature Scaling
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Dimensionality Reduction with Truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)  # You can adjust n_components based on your specific dataset and needs
train_features_reduced = svd.fit_transform(train_features_scaled)
test_features_reduced = svd.transform(test_features_scaled)

# Classifier Training
classifier = SVC(kernel='rbf', gamma='scale')
classifier.fit(train_features_reduced, train_labels)
predictions = classifier.predict(test_features_reduced)

# Classification Report
print(classification_report(test_labels, predictions))