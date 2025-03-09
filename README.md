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
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn skorch imblearn
```

### 3. Execute Notebooks

- Run notebooks directly using **Jupyter Notebook** or **Google Colab**.
- Ensure **CUDA** is available to leverage GPU acceleration for optimal performance.

---

## 👨‍💻 **Author**
- **Mohamed BARBYCH**  
- **FST Tangier, UAE, Morocco**  



## 🚀 Part Two: Multi-Class Classification (Predictive Maintenance)

### 📌 **Objective**
This section implements a **Multi-Class Classification Model** using **PyTorch** to predict **machine failures** based on various sensor readings. The model is trained using a **fully connected Deep Neural Network (DNN)** and evaluated using **accuracy, F1-score, and regularization techniques**.

---

## 📂 **Dataset:**
We use the **Predictive Maintenance Dataset** from [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification), which contains **sensor measurements** and failure classifications.

### 🔹 **Dataset Columns:**
- `UDI` (Unique Identifier) → **Dropped**
- `Product ID` → **Categorical (Encoded)**
- `Type` → **Categorical (Encoded)**
- `Air temperature [K]` → **Numerical**
- `Process temperature [K]` → **Numerical**
- `Rotational speed [rpm]` → **Numerical**
- `Torque [Nm]` → **Numerical**
- `Tool wear [min]` → **Numerical**
- `Target` → **Binary Target (Failure: 1, No Failure: 0)**
- `Failure Type` → **Multi-Class Target (Categorical, Encoded)**

---

## 🔎 **1. Data Preprocessing**
Before training the model, we **clean and preprocess the dataset**:

✔ **Dropped unnecessary columns** (`UDI`, `Product ID`)  
✔ **Encoded categorical variables** (`Type`, `Failure Type`)  
✔ **Standardized numerical features** using `StandardScaler`  
✔ **Applied SMOTE** to handle class imbalance  

💡 **Why these steps?**
- **Standardization** improves training stability.
- **SMOTE (Synthetic Minority Over-sampling Technique)** ensures balanced training across failure types.

---

## 📊 **2. Exploratory Data Analysis (EDA)**
- **Visualized class distribution** to detect imbalances.
- **Analyzed feature correlations** to understand relationships.

---

## 🛠 **3. Data Augmentation (Class Balancing)**
To handle class imbalance, we used **SMOTE (Synthetic Minority Over-sampling Technique)**, which **generates synthetic samples** to ensure that each failure type is adequately represented.

💡 **Why SMOTE?**
- Prevents bias toward majority classes.
- Helps the model generalize better.

---

## 🏗 **4. Deep Neural Network (DNN) Architecture**
We implemented a **3-layer fully connected network**:

✔ **Input Layer:** Accepts **all sensor features**.  
✔ **Hidden Layers:** **ReLU activation** for non-linearity.  
✔ **Output Layer:** Uses **Softmax activation** for **multi-class classification**.

💡 **Why ReLU Activation?**
- Faster training convergence.
- Helps avoid vanishing gradients.

---

## ⚙️ **5. Hyperparameter Tuning**
To optimize the model, we used **GridSearchCV** to find the best:
- **Learning Rate** (`lr`)
- **Number of Hidden Layers**
- **Number of Neurons per Layer**
- **Optimizer** (`Adam`, `SGD`)
- **Number of Epochs**

🔹 **Best Found Hyperparameters:**
- `Hidden Layers: 64 → 32`
- `Learning Rate: 0.001`
- `Optimizer: Adam`
- `Epochs: 50`

💡 **Why GridSearchCV?**
- Automates hyperparameter tuning for better generalization.
- Ensures we use the most efficient model configuration.

---

## 📉 **6. Loss vs Epochs**
![Loss vs Epochs](path/to/loss_vs_epochs.png)

✔ **Loss is steadily decreasing**  
✔ **No overfitting** (Train and Test loss are close)

💡 **Interpretation:**  
The model is **learning properly**, as both **training and test loss decrease smoothly**. If the test loss were much higher than training loss, it would indicate **overfitting**.

---

## 📈 **7. Accuracy vs Epochs**
![Accuracy vs Epochs](path/to/accuracy_vs_epochs.png)

✔ **Train and Test Accuracy improve together**  
✔ **Test Accuracy reaches over 90%**  

💡 **Interpretation:**  
The model is **generalizing well**. Since **train and test accuracy curves are close**, we **do not observe overfitting**.

---

## 📊 **8. Performance Evaluation**
### **Classification Report**
```plaintext
Precision | Recall | F1-Score | Support
---------------------------------------
Class 0 | 0.95 | 0.92 | 0.93 | N samples
Class 1 | 0.90 | 0.91 | 0.90 | N samples
...
