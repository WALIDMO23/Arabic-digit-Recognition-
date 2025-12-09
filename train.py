"""
Arabic Digit Recognition - Compare Multiple ML Algorithms
This script compares various machine learning algorithms and saves the best model
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from PIL import Image, ImageOps
import cv2
from skimage.feature import hog, local_binary_pattern

# Import all ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# =============== CONFIGURATION ===============
DATA_DIR = "data"
IMG_SIZE = (64, 64)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature extraction settings
USE_HOG = True
USE_LBP = True
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 24

# =============== LOAD AND PROCESS IMAGES ===============
def load_images():
    """Load images from organized folders"""
    images = []
    labels = []
    class_names = []

    for label in range(10):
        label_dir = os.path.join(DATA_DIR, str(label))
        if os.path.exists(label_dir):
            class_names.append(str(label))
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(label_dir, file)
                    try:
                        # Load and preprocess
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img)

                        # Auto-invert if needed
                        if np.mean(img_array) < 128:
                            img = ImageOps.invert(img)
                            img_array = np.array(img)

                        # Resize
                        img_array = cv2.resize(img_array, IMG_SIZE)
                        images.append(img_array)
                        labels.append(label)
                    except:
                        continue

    return np.array(images), np.array(labels), class_names

# =============== FEATURE EXTRACTION ===============
def extract_features(images):
    """Extract features from images"""
    features = []

    for img in images:
        img_features = []

        # HOG features
        if USE_HOG:
            hog_feat = hog(
                img,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                channel_axis=None
            )
            img_features.extend(hog_feat)

        # LBP features
        if USE_LBP:
            lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
            hist = hist.astype('float32') / (hist.sum() + 1e-6)
            img_features.extend(hist)

        features.append(img_features)

    return np.array(features)

# =============== COMPARE ML ALGORITHMS ===============
def compare_algorithms(X_train, X_test, y_train, y_test):
    """Compare multiple ML algorithms"""

    # Define all algorithms to compare
    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'SVM (Linear)': SVC(kernel='linear', random_state=RANDOM_STATE, probability=True),
        'SVM (RBF)': SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE)
    }

    results = []

    for name, model in algorithms.items():
        print(f"\n⏳ Training {name}...")
        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Store results
        results.append({
            'Algorithm': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Training Time (s)': train_time,
            'Model': model
        })

        print(f"   ✅ Train Accuracy: {train_acc:.4f}")
        print(f"   ✅ Test Accuracy: {test_acc:.4f}")
        print(f"   ⏱️  Training Time: {train_time:.2f}s")

        # Print classification report for test set
        print(f"   📊 Classification Report:")
        print(classification_report(y_test, y_test_pred, digits=3))

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}\nTest Accuracy: {test_acc:.2%}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{name.replace(" ", "_")}.png', dpi=150)
        plt.close()

    return pd.DataFrame(results)

# =============== MAIN FUNCTION ===============
def main():
    print("=" * 70)
    print("ARABIC DIGIT RECOGNITION - ML ALGORITHMS COMPARISON")
    print("=" * 70)

    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load images
    print("\n📥 Loading images...")
    images, labels, class_names = load_images()

    if len(images) == 0:
        print("❌ No images found!")
        return

    print(f"✅ Loaded {len(images)} images")
    print(f"📊 Class distribution: {np.bincount(labels)}")

    # Extract features
    print("\n🔬 Extracting features...")
    X = extract_features(images)
    y = labels

    print(f"📊 Feature matrix shape: {X.shape}")

    # Split data
    print("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Compare algorithms
    print("\n" + "=" * 70)
    print("🤖 COMPARING MACHINE LEARNING ALGORITHMS")
    print("=" * 70)

    results_df = compare_algorithms(X_train_scaled, X_test_scaled, y_train, y_test)

    # Sort by test accuracy
    results_df = results_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(results_df[['Algorithm', 'Train Accuracy', 'Test Accuracy', 'Training Time (s)']].to_string(index=False))

    # Save results to CSV
    results_df.to_csv('results/algorithm_comparison.csv', index=False)
    print("\n💾 Results saved to: results/algorithm_comparison.csv")

    # =============== SAVE BEST MODEL ===============
    print("\n" + "=" * 70)
    print("🏆 SAVING BEST MODEL")
    print("=" * 70)

    # Get best model
    best_idx = results_df['Test Accuracy'].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_name = results_df.loc[best_idx, 'Algorithm']
    best_test_acc = results_df.loc[best_idx, 'Test Accuracy']

    print(f"✅ Best Algorithm: {best_name}")
    print(f"✅ Test Accuracy: {best_test_acc:.4f}")

    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')

    # Save model info
    model_info = {
        'algorithm': best_name,
        'test_accuracy': float(best_test_acc),
        'train_accuracy': float(results_df.loc[best_idx, 'Train Accuracy']),
        'training_time': float(results_df.loc[best_idx, 'Training Time (s)']),
        'feature_config': {
            'use_hog': USE_HOG,
            'use_lbp': USE_LBP,
            'img_size': IMG_SIZE,
            'hog_orientations': HOG_ORIENTATIONS
        },
        'class_names': class_names
    }

    joblib.dump(model_info, 'models/model_info.pkl')
    print(f"💾 Best model saved to: models/best_model.pkl")

    # =============== VISUALIZATION ===============
    print("\n" + "=" * 70)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Plot comparison chart
    plt.figure(figsize=(14, 8))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    bars = plt.barh(results_df['Algorithm'], results_df['Test Accuracy'], color='skyblue')
    plt.xlabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.xlim([0, 1])
    for bar, acc in zip(bars, results_df['Test Accuracy']):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}',
                va='center', fontsize=9)

    # Training time comparison
    plt.subplot(1, 2, 2)
    bars = plt.barh(results_df['Algorithm'], results_df['Training Time (s)'], color='lightcoral')
    plt.xlabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    for bar, time_val in zip(bars, results_df['Training Time (s)']):
        plt.text(time_val + 0.1, bar.get_y() + bar.get_height()/2, f'{time_val:.1f}s',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/algorithm_comparison_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Visualizations saved to 'results/' directory")

    # =============== DETAILED REPORT ===============
    print("\n" + "=" * 70)
    print("📋 GENERATING DETAILED REPORT")
    print("=" * 70)

    # Generate detailed report for best model
    y_pred = best_model.predict(X_test_scaled)

    print(f"\n🏆 BEST MODEL DETAILS: {best_name}")
    print(f"   Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print(f"   Train Accuracy: {results_df.loc[best_idx, 'Train Accuracy']:.4f}")

    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    print("\n🔢 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save detailed report
    with open('results/detailed_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ARABIC DIGIT RECOGNITION - DETAILED REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("BEST MODEL:\n")
        f.write(f"  Algorithm: {best_name}\n")
        f.write(f"  Test Accuracy: {best_test_acc:.4f}\n")
        f.write(f"  Train Accuracy: {results_df.loc[best_idx, 'Train Accuracy']:.4f}\n\n")

        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

        f.write("\nCONFUSION MATRIX:\n")
        f.write(str(cm))

        f.write("\n\nALL ALGORITHMS COMPARISON:\n")
        f.write(results_df.to_string(index=False))

    print("\n✅ Detailed report saved to: results/detailed_report.txt")
    print("\n🎉 COMPARISON COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()