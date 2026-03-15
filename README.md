# 🫁 TB-Vision

### Explainable AI for Tuberculosis Screening in Resource-Limited Settings

TB-Vision is a **clinical decision support system for tuberculosis screening** that combines lightweight deep learning models with explainable AI and intelligent validation. 
The system is designed for **rural clinics and low-resource healthcare environments**, where radiologists and diagnostic infrastructure are limited.

Instead of relying solely on cloud AI, TB-Vision follows an **offline-first architecture**:

1️⃣ **Local CNN ensemble** analyzes chest X-rays  
2️⃣ **Uncertainty estimation** determines prediction confidence  
3️⃣ **Cloud validation** is triggered only for uncertain cases  

This hybrid design enables **fast, affordable, and scalable TB screening worldwide.**

---

### 🔗 Links

| Resource | Link |
|--------|------|
| Live Demo | https://tbvision.vercel.app |
| Demo Video | https://youtu.be/... |
| GitHub Repository | https://github.com/bibekjoshi01/TBVision |

---

### 🧠 Core Idea

Most AI systems for medical imaging suffer from:

- black-box predictions
- overconfident outputs
- lack of clinical context
- dependence on cloud infrastructure

TB-Vision solves these problems through:

- **Explainable AI (Grad-CAM++)**
- **Uncertainty-aware predictions**
- **Offline-first deployment**
- **Multi-stage AI validation**
