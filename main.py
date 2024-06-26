"""
Fox recognition using SIFT and SVD
"""
import os
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


def load_images_and_labels(base_dir, shuffle_data=False):
    """
    Load images and categorise it into np array with corresponding label array,
    Resize images to 128x128
    Suffle the array if specified as shuffle_data=True

    Args:
        base_dir (str): path to directory
        shuffle_data (bool, optional): flag if shuffling the array is needed. Defaults to False.

    Returns:
        np.array, np.array: first array is for images, second for labels
    """
    images = []
    labels = []
    categories = {
            0: 'not_fox',
            1: 'fox'
    }
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
    """
    In this function we extract key descriptors using SIFT algorithm
    Each image produces different number of keypoints and descriptors

    Args:
        images (np.array): np.array of images

    Returns:
        list[]: list of descriptors, concatenated vectors of descriptors
        Each descriptor is a 128-element vector
    """
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray_img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.zeros((0, 128)))
    return descriptors_list


def train_kmeans(descriptors_list, n_clusters=250):
    """This function transforms all the feature vectors to have the same length

    Args:
        descriptors_list (list): list of descriptors
        n_clusters (int, optional): The number of clusters. Defaults to 250.

    Returns:
        list: list of normalized feature vectors
    """
    all_descriptors = np.vstack([desc for desc in descriptors_list if desc.size > 0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans


def create_histograms(descriptors_list, kmeans):
    """function for creating histograms

    Args:
        descriptors_list (list): list of descriptors
        kmeans (list): list of normalized feature vectors

    Returns:
        np.array: list of normalized feature vectors
    """
    histograms = []
    for descriptors in descriptors_list:
        if descriptors.size > 0:
            labels = kmeans.predict(descriptors)
            hist = np.histogram(labels, bins=np.arange(kmeans.n_clusters+1))[0]
        else:
            hist = np.zeros(kmeans.n_clusters)
        histograms.append(hist)
    return np.array(histograms)


def main():
    """
    main function
    """
    train_images, train_labels = load_images_and_labels('./test',  shuffle_data=True)
    test_images, test_labels = load_images_and_labels('./train', shuffle_data=True)

    train_descriptors = extract_sift_descriptors(train_images)
    test_descriptors = extract_sift_descriptors(test_images)

    kmeans = train_kmeans(train_descriptors)

    train_features = create_histograms(train_descriptors, kmeans)
    test_features = create_histograms(test_descriptors, kmeans)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    svd = TruncatedSVD(n_components=20, random_state=42)
    train_features_reduced = svd.fit_transform(train_features_scaled)
    test_features_reduced = svd.transform(test_features_scaled)

    classifier = SVC(kernel='rbf', gamma='scale')
    classifier.fit(train_features_reduced, train_labels)
    predictions = classifier.predict(test_features_reduced)

    print(classification_report(test_labels, predictions))

if __name__ == "__main__":
    main()
