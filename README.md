# Applied-AI-Project


# Indoor/Outdoor Museum Scene Classification (CNN vs. Traditional ML)

## Project Overview

This project explores and compares various machine learning and deep learning techniques for classifying images of museum scenes as either 'indoor' or 'outdoor'. It implements and evaluates:

1.  A custom Convolutional Neural Network (CNN) designed for this task.
2.  Traditional Machine Learning Models:
    *   Decision Trees (Supervised)
    *   Decision Trees (Semi-Supervised using Pseudo-Labeling)
    *   Random Forest
    *   XGBoost

The primary goal is to assess the effectiveness of these different approaches, particularly the CNN, on a specific subset of the MIT Places dataset resized to a lower resolution (64x64 pixels). Performance is evaluated using standard metrics: Accuracy, Precision, Recall, and F1-score.

## Key Findings

*   The custom **Convolutional Neural Network (CNN) significantly outperformed** all traditional machine learning models tested.
*   The **best CNN configuration achieved 93% accuracy** on the test set (trained with LR=0.0001 for 10 epochs).
*   Traditional ensemble models (Random Forest, XGBoost) achieved accuracies around 83-84%.
*   Decision Tree models performed lowest, around 75% accuracy.
*   CNN performance was highly sensitive to hyperparameter tuning, especially the learning rate and number of training epochs.

## Dataset

*   **Source:** Derived from the [MIT Places Dataset](http://places.csail.mit.edu/) ([1, 2]).
*   **Content:** Images specifically curated to represent museum indoor and outdoor scenes.
*   **Structure:**
    *   **Training Set:** 10,000 images (5,000 indoor, 5,000 outdoor). File paths listed in `train.txt`.
    *   **Test Set:** 200 images (100 indoor, 100 outdoor). File paths listed in `val.txt`.
*   **Preprocessing:** All images are resized to **64x64 pixels**.
*   **Labels:** Derived from the directory structure.

*Note: The actual image data is not included in this repository. You will need to obtain the MIT Places dataset and structure it according to the paths specified in `train.txt` and `val.txt`.*

## Methodology

### 1. Convolutional Neural Network (CNN)

*   **Model:** A custom CNN architecture (`SimpleCNN`) built using PyTorch.
    *   Input: 3x64x64 images.
    *   Layers: 2 Convolutional blocks (Conv2D -> ReLU -> MaxPool2D) followed by a Flatten layer and a fully connected output layer.
    *   No Batch Normalization or Dropout layers were used in this specific implementation.
*   **Preprocessing:** Images resized to 64x64, channels ordered (CHW), pixel values normalized to [0.0, 1.0].
*   **Training:**
    *   Optimizer: Adam
    *   Loss: Cross-Entropy Loss
    *   Hyperparameters (varied):
        *   Learning Rate (LR): 0.01, 0.001, 0.0001
        *   Epochs: 10, 20
        *   Batch Size: 32 (fixed)
*   **Note:** The provided code performs a single training run with fixed hyperparameters and evaluates directly on the test set (`val.txt`). No validation split or early stopping was implemented in the script used for generating the reported results.

### 2. Traditional Machine Learning Models

*   **Features:** RGB Color Histograms (3 channels x 256 bins = 768 dimensions) extracted using OpenCV after resizing images to 64x64.
*   **Models Evaluated:**
    *   Decision Tree (max_depth=10)
    *   Random Forest (100 trees)
    *   XGBoost
    *   Semi-Supervised Decision Tree (using pseudo-labeling)
*   **Framework:** Scikit-learn, XGBoost library. Models saved/loaded using joblib.

## Performance Highlights (Best Models)

**CNN (LR=0.0001, 10 Epochs):**

*   **Accuracy: 0.93**
*   Classification Report:
  ```
                    precision    recall  f1-score   support
    museum-indoor       0.92      0.93      0.93       100
   museum-outdoor       0.93      0.92      0.92       100
         accuracy                           0.93       200
        macro avg       0.93      0.93      0.93       200
     weighted avg       0.93      0.93      0.93       200
```
**XGBoost:**

*   Accuracy: ~0.84
*   F1-Score (Macro Avg): ~0.835

**Random Forest:**

*   Accuracy: ~0.83
*   F1-Score (Macro Avg): ~0.83


## Project Structure (Example)

```
your-repo-name/  
├── train/ # Directory containing training images  
│ ├── museum-indoor/  
│ │ ├── image_train_in_001.jpg  
│ │ └── ...  
│ └── museum-outdoor/  
│ ├── image_train_out_001.jpg  
│ └── ...  
├── val/ # Directory containing test/validation images  
│ ├── museum-indoor/  
│ │ ├── image_val_in_001.jpg  
│ │ └── ...  
│ └── museum-outdoor/  
│ ├── image_val_out_001.jpg  
│ └── ...  
├── CNN.ipynb 
├── DecisionTree.ipynb  implementation  
├── RandomForest.ipynb implementation  
├── Semi-Supervised DT.ipynb   
├── boosting_xgboost.ipynb  
├── train.txt 
├── val.txt 
└── README.md
```

## Contributors

*   Hritik Nandanwar (Student ID: 40268450)
*   Zeel Ketan Divawala (Student ID: 40272584)
*   Soham Sakaria (Student ID: 40275636)

*Affiliation: Gina Cody School of Engineering, Concordia University*

## References

[1] B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva, "Learning deep features for scene recognition using places database," in *Advances in Neural Information Processing Systems (NIPS)*, 2014, vol. 27.

[2] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, "Places: A 10 million image database for scene recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 6, pp. 1452–1464, June 2018.
