
#  Heart Attack Risk Detector (Neural Network from Scratch)

This project implements a **Heart Attack Risk Detector** using a neural network built entirely from scratch in Python with **NumPy**. The model is trained to predict whether a person is at risk of a heart attack based on clinical and physiological features.

The network was built by studying the guide:  
[Learn to Build a Neural Network from Scratch – Yes, Really!](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc)

---

##  Features

- Neural Network implemented **without** high-level ML libraries
-  Uses forward propagation, backpropagation, and gradient descent
-  Handles preprocessing like:
  - Min-max normalization for numerical data
  - One-hot encoding if applicable
-  Binary classification:  
  - `0` → Not at risk  
  - `1` → At risk
-  Accuracy tracking across epochs

---

##  Dataset

The model is trained and tested on a labeled CSV dataset of patient health records.

###  Format:
```

Age,SysBP,Cholesterol,Risk

```

- Most columns are numeric or categorical heart-health indicators.
- `target`:  
  - `1` = Patient is at risk of heart attack  
  - `0` = Patient is not at risk

---

##  Project Structure

```

HeartAttackRiskDetector/
├── trained\_model.npz            # Saved model weights after training
├── training\_data/
│   └── training\_data.csv        # CSV dataset used for training
│
└── scripts/                     # All source code
├── **init**.py
├── DataUtils.py             # Preprocessing: normalization, loading
├── NeuralNetwork.py         # Neural network class (from scratch)
├── Train.py                 # Train the model
├── Trainer.py               # Training utilities (batching, metrics)
├── Predict.py               # Predict single samples from CLI
└── Predictor.py             # Batch predictions or test set evaluation

````

---

##  Neural Network Architecture

| Layer        | Description                                       |
|--------------|----------------------------------------------------|
| Input Layer  | Number of features (after preprocessing)           |
| Hidden Layers| Configurable hidden layers (e.g. [32, 16])         |
| Output Layer | 1 unit with **Sigmoid** activation (risk score)    |
| Activation   | **ReLU** for hidden layers, **Sigmoid** for output |
| Loss         | **Binary Cross Entropy**                           |
| Optimizer    | **Gradient Descent** (manual, from scratch)        |

---

## 🛠 How to Run

### 1. Install dependencies
```bash
pip install numpy pandas
````

### 2. Train the model

```bash
python scripts/Train.py
```

### 3. Predict on a new sample

```bash
python scripts/Predict.py
```

### 4. Run batch predictions or evaluate test set

```bash
python scripts/Predictor.py
```

---

##  Key Learnings

* How neural networks work **under the hood**
* Manual implementation of forward & backward propagation
* Preprocessing techniques for structured health data
* Basics of binary classification for health risk prediction

---

##  Tech Stack

* **Language**: Python
* **Libraries**: NumPy, Pandas
* **No external ML frameworks used**

---

##  License

This project is created for educational purposes.
You are welcome to use, adapt, and build upon it.




