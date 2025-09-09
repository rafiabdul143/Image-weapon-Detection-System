🔥🔫 Image Weapon and Fire Detection System using NASA Net
📌 Overview

This project implements a real-time image classification system that detects weapons (e.g., guns, knives) and fire hazards using the NASA Net algorithm (Xception-based CNN). The system is designed for security monitoring and public safety, where quick and accurate detection is crucial.

By leveraging deep learning with NASA Net, the system achieves high accuracy while maintaining efficient performance, making it suitable for edge devices, CCTV surveillance, and mobile platforms.

🚀 Features

Detects weapons (guns, knives, etc.) from images and video frames.

Detects fire and flames for hazard prevention.

NASA Net-based model (Xception CNN) for efficient and accurate detection.

Can be deployed on Raspberry Pi, Jetson Nano, or any edge device.

Supports real-time video stream processing.

Generates alerts/notifications when a weapon or fire is detected.

🧠 Algorithm – NASA Net (Xception CNN)

NASA Net is built on the Xception architecture, a deep convolutional neural network optimized for high-accuracy image classification.

Uses depthwise separable convolutions to significantly reduce computation while improving accuracy.

Employs transfer learning from pretrained ImageNet weights, fine-tuned for weapon and fire detection datasets.

Suitable for real-time applications due to its balance of speed and performance.

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

Add YOLOv8 or EfficientNet for advanced detection.

Deploy as a Flask/Django web app or mobile app.

Integrate with IoT devices for automated alarms.

Extend to multi-class dangerous object detection.

📖 Documentation

Complete project documentation (including system design, methodology, datasets used, training process, and results) is available inside the docs/ folder as a PDF.
This documentation can be used for academic submission or as a developer guide.

🤝 Contributing

Contributions are welcome! Please fork this repo and submit a pull request.
