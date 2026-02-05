# Handwritten Digit Recognition with CNN

**Abdulhamid Alaboud · Ahmed Alaglan**  
Department of Computer Science · Jazan University

## Project Overview
A convolutional neural network implementation for handwritten digit classification using the MNIST dataset. The model achieves **99.0% test accuracy** through automated feature learning from raw pixel data.

## Key Features
- **Complete Implementation**: Full workflow from data loading to model evaluation
- **LeNet-Inspired Architecture**: Efficient CNN design optimized for digit recognition
- **Comprehensive Analysis**: Confusion matrix, classification metrics, and visualizations
- **Well-Documented**: Clear code structure with detailed explanations

## Results
| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.0% |
| **Precision (Avg)** | 99.0% |
| **Recall (Avg)** | 99.0% |
| **F1-Score (Avg)** | 99.0% |

## Repository Structure
```
Handwritten-Digit-Recognition-CNN/
├── Handwritten-Digit-Recognition-CNN.ipynb  # Complete CNN implementation
├── README.md                                 # Project documentation
├── LICENSE                                   # MIT License
├── .gitignore                                # Git exclusion rules
├── requirements.txt                          # Python dependencies
└── .project_version                          # Project version (v1.0.0)
```

## Quick Start
```bash
# 1. Clone repository
git clone https://github.com/Ahmed-Alaglan/Handwritten-Digit-Recognition-CNN.git
cd Handwritten-Digit-Recognition-CNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook Handwritten-Digit-Recognition-CNN.ipynb
```

## Dependencies
```txt
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
pandas>=1.5.0
jupyter>=1.0.0
```

## Model Architecture
```
Input (28×28×1) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
→ Flatten → Dense(128) → Dropout → Dense(10) → Output
```
*Total Parameters: ~225,000*

## Performance Analysis
- **Training Accuracy**: 99.8%
- **Validation Accuracy**: 99.3%
- **Test Accuracy**: 99.0%

## Dataset
- **MNIST**: 70,000 grayscale images (28×28 pixels)
- **Training**: 60,000 images
- **Testing**: 10,000 images
- **Classes**: Digits 0-9 (balanced distribution)

## Methodology
1. **Data Preparation**: Normalization & reshaping
2. **Model Design**: CNN with convolutional and pooling layers
3. **Training**: Adam optimizer, 10 epochs, batch size 128
4. **Evaluation**: Multiple metrics and visualization tools

## Notebook Contents
The main notebook includes:
- Data loading and exploration
- Model architecture definition
- Training and validation process
- Performance evaluation and visualization
- Sample predictions and analysis

## Applications
- Digital document processing
- Bank check recognition
- Educational AI tools
- Touchscreen input systems

## Future Enhancements
- Real-time web interface
- Mobile deployment (TensorFlow Lite)
- Ensemble methods for higher accuracy
- Transfer learning for custom datasets

## Citation
```bibtex
@software{alaboud2024mnist,
  title = {Handwritten Digit Recognition with CNN},
  author = {Alaboud, Abdulhamid and Alaglan, Ahmed},
  year = {2024},
  url = {https://github.com/Ahmed-Alaglan/Handwritten-Digit-Recognition-CNN}
}
```

## License
MIT License - See LICENSE file for details

## Contact
For questions or collaborations, open an issue in this repository.

---

*Built with TensorFlow/Keras | MNIST Dataset | Jazan University Research*

## Changelog
- **v1.0.0**: Initial release with complete CNN implementation and documentation
