# DeepLearning-PyTorch-NYSE-PredictiveMaintenance

## Short Description
Implementation of Deep Neural Networks for Regression and Multi-Class Classification using PyTorch on NYSE and Predictive Maintenance datasets.

---

## 📌 **Objective**
Learn and practice PyTorch by implementing Deep Neural Networks for:

- **Regression Task:** NYSE Dataset
- **Multi-Class Classification Task:** Predictive Maintenance Dataset

---

## 🚀 **Part One: Regression (NYSE Dataset)**

### **Dataset:**
[NYSE Fundamentals Dataset](https://www.kaggle.com/datasets/dgawlik/nyse)

### **Goals**:
- Exploratory Data Analysis (EDA)
- DNN architecture with PyTorch
- Hyperparameter tuning using GridSearchCV
- Visualization of Loss and Accuracy metrics
- Application of regularization techniques (Dropout & L2)

---

## ✅ **Steps Implemented**

### 1. **Exploratory Data Analysis (EDA)**
- Cleaned and standardized data.
- Analyzed feature correlations.
- Visualized important relationships.

### 2. **Regression Model**
- Developed Deep Neural Network using PyTorch.
- Trained the model with GPU (CUDA support).

### 3. **Hyperparameter Tuning**
- Optimized hyperparameters using GridSearchCV (`skorch`).
- Best hyperparameters identified: learning rate, optimizers, epochs, and hidden layers.

### 4. **Visualization**
- Loss vs. Epochs (Training & Test)
- Accuracy (R² Score) vs. Epochs (Training & Test)

### 5. **Regularization**
- Implemented Dropout & L2 regularization.
- Compared original vs regularized models' performance.

---

## 🚀 **Part Two: Multi-Class Classification**

**Dataset:** [Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

---

## ✅ **Steps Implemented**

### 1. **Data Preprocessing**
- Cleaned and handled missing values.
- Standardized/normalized features.
- Encoded categorical variables.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized class distribution.
- Performed correlation analysis.

### 3. **Data Augmentation**
- Applied SMOTE for balancing class distribution.

### 4. **Deep Neural Network (PyTorch)**
- Built Multi-Class classification neural network.
- Trained on GPU/CPU with CrossEntropyLoss.

### 4. **Hyperparameter Tuning (GridSearchCV)**
- Tuned learning rate, optimizer, epochs, and hidden layer architecture.

### 5. **Visualization**
- Loss vs. Epochs (Training & Test)
- Accuracy vs. Epochs (Training & Test)

### 6. **Evaluation Metrics**
- Computed Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

### 7. **Regularization**
- Implemented Dropout & L2 regularization.
- Compared results before and after regularization.

---

## ⚙️ **Technologies Used**
- Python
- PyTorch
- Scikit-learn (GridSearchCV)
- Skorch (PyTorch wrapper for sklearn compatibility)
- pandas, numpy, matplotlib, seaborn
- imblearn (for SMOTE)

---

## 🚀 **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone <repository-link>
