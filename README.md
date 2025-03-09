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
![image](https://github.com/user-attachments/assets/127306ee-f706-4934-970a-9c1285d276b6)
![image](https://github.com/user-attachments/assets/a5d4240a-b1ad-4ce6-9814-f689dc9a8aa6)


### 2. **Regression Model**
- Developed Deep Neural Network using PyTorch.
- Trained the model with GPU (CUDA support).

### 3. **Hyperparameter Tuning**
- Optimized hyperparameters using GridSearchCV (`skorch`).
- Best hyperparameters identified: learning rate, optimizers, epochs, and hidden layers.

### 4. **Visualization**
- Loss vs. Epochs (Training & Test)
- Accuracy (R² Score) vs. Epochs (Training & Test)
![image](https://github.com/user-attachments/assets/d928ce3d-f472-4b1f-87c0-d4605c75405b)

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
![image](https://github.com/user-attachments/assets/58132e73-ab2d-4caa-b57b-cb18f7992bf3)

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
![image](https://github.com/user-attachments/assets/b9443db6-acbf-4c53-8f17-a89d49acfe22)

✔ **Loss is steadily decreasing**  
✔ **No overfitting** (Train and Test loss are close)

💡 **Interpretation:**  
The model is **learning properly**, as both **training and test loss decrease smoothly**. If the test loss were much higher than training loss, it would indicate **overfitting**.

---

## 📈 **7. Accuracy vs Epochs**
![image](https://github.com/user-attachments/assets/cd3cd3dc-8b98-4316-8139-6a09613c4b0a)

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
```
---

## 💡 **Key Metrics Explained**
- **Accuracy**: Measures overall correctness.
- **Precision**: How many predicted failures were actual failures?
- **Recall**: How many actual failures did the model detect?
- **F1-Score**: Harmonic mean of Precision & Recall.
- ✅ **Final Model Accuracy: 92%**

---

## 🛑 **9. Regularization and Overfitting Prevention**
To **prevent overfitting**, we applied:
- **L2 Regularization** (`weight_decay=1e-4`)
- **Dropout (30%)** to randomly deactivate neurons.

### 🔹 **Results Comparison**
![image](https://github.com/user-attachments/assets/f4fb282e-8de5-4548-a14e-4fcada7f62a0)

### 💡 **Observations**
- The **regularized model achieves higher test accuracy**.
- The **original model overfits slightly**, while the **regularized model generalizes better**.

---

## 🎯 **Final Summary**

| Step                     | Key Findings                                      |
|--------------------------|--------------------------------------------------|
| **Data Cleaning**        | Dropped unused columns, encoded categories      |
| **EDA**                 | Balanced dataset with SMOTE                      |
| **DNN Model**           | Fully connected network with 2 hidden layers     |
| **Hyperparameter Tuning** | Found optimal learning rate, neurons, and optimizer |
| **Performance**         | 92% accuracy, strong F1-score                    |
| **Regularization**      | Reduced overfitting, improved generalization     |

🚀 **Conclusion:**  
This model effectively predicts **machine failures** with **high accuracy** and is **robust against overfitting**.  
**Regularization further enhances generalization.**

---

## 📌 **How to Run the Code**
```bash
# Install required dependencies
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn skorch imblearn

# Run the Python script
python train_maintenance_model.py
