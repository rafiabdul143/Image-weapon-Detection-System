🔥🔫 Image Weapon and Fire Detection System using MobileNet
📌 Overview

This project implements a real-time image classification system that detects weapons (e.g., guns, knives) and fire hazards using the MobileNet algorithm. The system is designed for security monitoring and public safety, where quick and accurate detection is crucial.

By leveraging deep learning with MobileNet, the system achieves high accuracy while remaining lightweight, making it suitable for edge devices, CCTV surveillance, and mobile platforms.

🚀 Features

Detects weapons (guns, knives, etc.) from images and video frames.

Detects fire and flames for hazard prevention.

MobileNet-based model for lightweight and efficient performance.

Can be deployed on Raspberry Pi, Jetson Nano, or any edge device.

Supports real-time video stream processing.

Generates alerts/notifications when a weapon or fire is detected.

🧠 Algorithm – MobileNet

MobileNet is a CNN architecture optimized for mobile and embedded vision applications.

Uses depthwise separable convolutions to reduce computation and model size.

Provides a balance between speed and accuracy for detection tasks.

Pretrained weights (on ImageNet) are fine-tuned for weapon and fire detection datasets.

📊 Dataset

Public datasets for weapons and fire detection were used.

Additional custom datasets can be added for improved accuracy.

Images are preprocessed (resized to 224×224, normalized).

📌 Use Cases

Smart Surveillance in public places (airports, malls, schools).

Fire hazard monitoring in industries or smart homes.

Security systems with real-time alerts.

Military & defense applications for weapon detection.

🔮 Future Enhancements

Add YOLOv8 or EfficientNet for better detection accuracy.

Deploy as a Flask/Django web app or mobile app.

Integrate with IoT devices for automatic alarms.

Support for multi-class dangerous object detection.

🤝 Contributing

Contributions are welcome! Please fork this repo and submit a pull request.
