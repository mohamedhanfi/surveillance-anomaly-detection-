# 🧠 Surveillance Anomaly Detection

## 📌 Overview

**Surveillance Anomaly Detection** is a robust, AI-powered system that automatically identifies abnormal or suspicious behavior in surveillance videos. Examples include incidents such as violence, accidents, theft, or any activity that deviates from typical patterns. The project aims to enhance intelligent video monitoring by offering a scalable, automated, and efficient anomaly detection solution.

It utilizes the large-scale **UCF-Crime** dataset, which contains real-world surveillance footage with labeled normal and anomalous events. To achieve high accuracy and reliability, we fine-tune cutting-edge video classification models like **SlowFast** and **MViT**, which are well-suited for capturing both temporal and spatial features in video sequences.

The system is modular and adaptable, supporting both real-time and offline (batch) processing. This makes it suitable for integration into a wide range of smart surveillance infrastructures where quick and accurate anomaly detection is essential.

---

## 🔧 Key Features

* 🚨 Detects suspicious behaviors such as violence, theft, and accidents in real-world surveillance footage
* 🎥 Trained on the UCF-Crime dataset: a diverse collection of annotated videos with normal and abnormal activities
* 🧠 Powered by advanced architectures: SlowFast for multi-speed feature processing and MViT for multi-scale visual understanding
* ⚙️ Modular pipeline: easy to integrate into existing video systems, adaptable for different deployment settings
* 🎞️ Supports both frame-level and clip-level analysis with efficient preprocessing and feature extraction
* 📈 Evaluation metrics: precision, recall, F1-score, accuracy, and AUC for robust performance assessment
* 📝 **Automated Reporting**: Uses OpenAI API to generate concise, human-readable reports for detected anomalies, including context like time, location, and confidence level

---

## 🛠️ Technology Stack

* **Python 🐍** — Core language for development
* **PyTorch 🔥** — Framework for deep learning and model training
* **PyTorchVideo 🎞️** — Tools and prebuilt models for video analysis
* **OpenCV 📷** — Video I/O and frame processing
* **Jupyter Notebooks 📒** — For experimentation, visualization, and documentation
* **OpenAI API 🧾** — For generating natural language summaries of detection events

---

## 🌍 Use Cases

* 🛡️ Public safety monitoring and emergency response
* 📹 Intelligent surveillance with reduced human supervision
* 🌆 Real-time anomaly detection for smart cities
* 🏥 High-security environments such as hospitals, banks, and campuses

---

## ✍️ Example Detection Report

Below is a sample output generated using OpenAI API based on a detected anomaly:

```text
Anomaly Detected: Violence
Location: Main Gate
Timestamp: 2025-05-04T15:30:12
Confidence: 92%

Report: A violent event was detected near the main gate on May 4th at 3:30 PM with high confidence. Immediate review is recommended to ensure public safety.
```

---

## 🤝 Contributing

We warmly welcome contributions, issue reports, suggestions, and collaborations! Feel free to fork the repository, submit pull requests, or open discussions to help improve and expand this project.
