# Exoplanet Detection Platform

**NASA Space Apps Challenge 2025 - A World Away: Hunting for Exoplanets**

A machine learning platform that automatically identifies exoplanets from astronomical data using a hierarchical ensemble of advanced models.

## Challenge Overview

**Challenge**: A World Away: Hunting for Exoplanets with AI  
**Summary**: Data from several different space-based exoplanet surveying missions have enabled discovery of thousands of new planets outside our solar system, but most of these exoplanets were identified manually. With advances in artificial intelligence and machine learning (AI/ML), it is possible to automatically analyze large sets of data collected by these missions to identify exoplanets.

**Our Solution**: A comprehensive hierarchical ensemble system that combines XGBoost, Random Forest, and Neural Networks to achieve state-of-the-art exoplanet detection.

## Features

### Advanced Machine Learning Architecture
- **Hierarchical Ensemble Model**: XGBoost + Random Forest → Neural Network
- **91.89% Accuracy**: Superior performance on NASA exoplanet datasets
- **95.41% AUC Score**: Excellent discrimination between exoplanets and false positives
- **96.43% Base Model Agreement**: High confidence in predictions

### Interactive Web Platform
- **Real-time Exoplanet Prediction**: Analyze individual celestial bodies with confidence scores
- **Batch Processing**: Upload CSV files for multiple object analysis
- **Interactive Visualizations**: Explore data patterns and model insights
- **Model Performance Metrics**: View accuracy, feature importance, and model statistics
- **Hyperparameter Tuning**: Interactive neural network configuration
- **Educational Content**: Learn about exoplanets and detection methods

### Comprehensive Analysis Tools
- **Correlation Matrix Visualization**: Understand feature relationships
- **Feature Importance Analysis**: Identify key planetary characteristics
- **ROC Curve Analysis**: Evaluate model discrimination performance
- **Precision-Recall Analysis**: Optimize for scientific confidence
- **Model Comparison**: Compare base models and ensemble performance

## Technical Architecture

### Hierarchical Ensemble System
```
Raw Features → [XGBoost, Random Forest] → Neural Network → Final Prediction
```

1. **XGBoost**: High-performance gradient boosting for base predictions
2. **Random Forest**: Feature selection and interpretability for base predictions
3. **Wide Feedforward Neural Network**: Meta-learner that combines base model outputs
4. **Hierarchical Structure**: Base models → Neural Network → Final prediction

### Data Pipeline
1. **Data Preprocessing**: Robust feature engineering and normalization
2. **Feature Selection**: 14 key planetary and stellar parameters (no data leakage)
3. **Model Training**: Hierarchical ensemble with proper train/test split
4. **Prediction**: Real-time classification with confidence scores

## Dataset and Features

### NASA Datasets Used
- **NASA Exoplanet Archive (Cumulative)**: `cumulative_2025.10.02_16.12.45.csv`
- **K2 Mission Data**: `k2pandc_2025.10.02_16.12.57.csv`
- **Total Training Samples**: 13,568 exoplanet candidates

### Key Features (14 Parameters)
#### Orbital Parameters
- `koi_period`: Orbital Period (days)
- `koi_time0bk`: Time of first transit (BJD - 2454833)
- `koi_impact`: Impact parameter
- `koi_duration`: Transit duration (hours)

#### Transit Characteristics
- `koi_depth`: Transit depth (ppm)
- `koi_prad`: Planetary radius (Earth radii)
- `koi_teq`: Equilibrium temperature (K)
- `koi_insol`: Insolation flux (Earth units)

#### Stellar Properties
- `koi_slogg`: Stellar surface gravity
- `koi_slogg_err1`, `koi_slogg_err2`: Stellar surface gravity errors
- `koi_srad`: Stellar radius (Solar radii)
- `koi_srad_err1`, `koi_srad_err2`: Stellar radius errors

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
```bash
# Clone or download the project
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt

# Launch the application
python launch_flask_app.py
```

The web interface will be available at `http://localhost:8080`

### Alternative Launch Methods
```bash
# Direct Flask launch
python flask_app.py

# Launch with model building visualization
python restart_with_training.py
```

## Project Structure

```
NasaSpaceApps2/
├── README.md                                    # This comprehensive guide
├── requirements.txt                             # Python dependencies
├── launch_flask_app.py                         # Application launcher
├── flask_app.py                                # Main Flask web application
├── hierarchical_ensemble_classifier.py         # Core ML classifier
├── robust_hierarchical_training.py            # Model building system
├── restart_with_training.py                   # Visual model building launcher
│
├── Data Files/
├── cumulative_2025.10.02_16.12.45.csv         # NASA Exoplanet Archive data
├── k2pandc_2025.10.02_16.12.57.csv           # K2 mission data
├── test_batch_data_20251004_151011.csv        # Sample test data
├── robust_feature_columns.txt                 # Feature definitions
│
├── Model Files/
├── robust_xgboost_model.pkl                   # Trained XGBoost model
├── robust_random_forest_model.pkl             # Trained Random Forest model
├── robust_hierarchical_ensemble.h5            # Trained neural network
├── robust_scaler.pkl                          # Data scaler
│
├── Web Interface/
├── templates/                                 # HTML templates
│   ├── index.html                            # Home page
│   ├── predict.html                          # Single prediction
│   ├── batch.html                            # Batch analysis
│   ├── statistics.html                       # Model statistics
│   ├── hyperparameters.html                  # Model tuning
│   └── learn.html                            # Educational content
│
├── static/                                   # Static assets
│   ├── css/                                 # Stylesheets
│   ├── js/                                  # JavaScript modules
│   └── images/                              # Images and icons
│
└── Documentation/
    ├── SpaceApps.docx                        # Project documentation
    └── Plan.pdf                             # Project plan
```

## Usage Guide

### Web Interface
1. **Single Prediction**: Enter planetary parameters manually or use templates
2. **Batch Analysis**: Upload CSV files with multiple objects for analysis
3. **Model Statistics**: View comprehensive performance metrics and visualizations
4. **Hyperparameter Tuning**: Experiment with neural network configurations
5. **Educational Content**: Learn about exoplanets and our detection methods

### Programmatic Usage
```python
from hierarchical_ensemble_classifier import HierarchicalEnsembleClassifier

# Initialize the classifier
classifier = HierarchicalEnsembleClassifier()
classifier.load_models()

# Prepare your data
data = {
    'koi_period': 365.0,
    'koi_time0bk': 1000.0,
    'koi_impact': 0.5,
    'koi_duration': 8.0,
    'koi_depth': 1000.0,
    'koi_prad': 1.0,
    'koi_teq': 300.0,
    'koi_insol': 1.0,
    'koi_slogg': 4.5,
    'koi_slogg_err1': 0.1,
    'koi_slogg_err2': -0.1,
    'koi_srad': 1.0,
    'koi_srad_err1': 0.05,
    'koi_srad_err2': -0.05
}

# Make prediction
result = classifier.classify_exoplanet(data)
print(f"Prediction: {'Exoplanet' if result['is_exoplanet'] else 'Not Exoplanet'}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"XGBoost Probability: {result['xgb_probability']:.2%}")
print(f"Random Forest Probability: {result['rf_probability']:.2%}")
```

## Model Performance

### Key Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Hierarchical Accuracy** | **91.89%** | Overall classification accuracy |
| **Hierarchical AUC** | **95.41%** | Area under ROC curve |
| **XGBoost Accuracy** | **91.89%** | Base model performance |
| **Random Forest Accuracy** | **91.53%** | Base model performance |
| **Base Model Agreement** | **96.43%** | Consensus between base models |
| **Precision** | **77.83%** | True positive rate |
| **Recall** | **83.79%** | Sensitivity |
| **F1-Score** | **80.70%** | Harmonic mean of precision and recall |

### Architecture Benefits
- **Hierarchical Learning**: Neural network effectively combines base model outputs
- **High Agreement**: 96.43% consensus between XGBoost and Random Forest
- **No Data Leakage**: Proper feature selection and data preprocessing
- **Robust Performance**: Excellent accuracy and discrimination ability

## Scientific Impact

This platform addresses the NASA Space Apps Challenge by:

1. **Automating Exoplanet Detection**: Reduces manual analysis time from hours to seconds
2. **Accelerating Discovery**: Identifies promising candidates faster than traditional methods
3. **Supporting Research**: Provides tools for astronomers and researchers worldwide
4. **Open Source**: Contributes to the scientific community with reproducible results
5. **Advanced Architecture**: Demonstrates state-of-the-art ensemble learning techniques

## Key Innovations

### Technical Innovations
- **Hierarchical Ensemble Architecture**: Novel combination of tree-based and neural network models
- **Automatic Model Building**: Fresh model training on every startup for optimal performance
- **Interactive Hyperparameter Tuning**: Real-time experimentation with neural network configurations
- **Comprehensive Visualization**: Advanced analytics and performance metrics

### Scientific Contributions
- **No Data Leakage**: Rigorous feature selection prevents overfitting
- **High Accuracy**: 91.89% accuracy on NASA exoplanet datasets
- **Interpretable Results**: Feature importance analysis for scientific understanding
- **Scalable Architecture**: Handles both single predictions and batch processing

## User Interface Features

### Modern Design
- **Dark/Light Mode**: Adaptive theme switching
- **Glassmorphism Effects**: Modern UI with liquid glass transparency
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Visualizations**: Chart.js integration for dynamic charts

### User Experience
- **Real-time Feedback**: Immediate prediction results with confidence scores
- **Batch Processing**: Upload CSV files for multiple object analysis
- **Template System**: Pre-configured parameters for common exoplanet types
- **Educational Content**: Learn about exoplanets and detection methods

## Dependencies

### Core ML Libraries
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.1.0` - Machine learning algorithms
- `xgboost>=1.6.0` - Gradient boosting framework
- `tensorflow>=2.10.0` - Deep learning framework
- `keras>=2.10.0` - High-level neural network API

### Web Framework
- `flask>=2.0.0` - Web application framework
- `flask-cors>=3.0.0` - Cross-origin resource sharing

### Visualization
- `matplotlib>=3.5.0` - Plotting library
- `seaborn>=0.11.0` - Statistical data visualization
- `plotly>=5.0.0` - Interactive plotting

### Utilities
- `joblib>=1.2.0` - Model persistence
- `werkzeug>=2.0.0` - WSGI utilities

## Future Enhancements

### Planned Features
- **Real-time Data Integration**: Connect to live astronomical data feeds
- **Advanced Ensemble Methods**: Implement stacking and blending techniques
- **Multi-mission Support**: Extend to TESS, JWST, and other missions
- **Collaborative Features**: Allow researchers to share and compare models

### Research Opportunities
- **Feature Engineering**: Develop new features from raw photometric data
- **Uncertainty Quantification**: Implement Bayesian methods for confidence intervals
- **Transfer Learning**: Adapt models to new missions and instruments
- **Explainable Models**: Enhance interpretability for scientific understanding

## Contributing

This project was developed for the NASA Space Apps Challenge 2025. We welcome contributions and improvements from the scientific community!

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Areas for Contribution
- **Model Improvements**: Enhance accuracy and performance
- **New Features**: Add functionality for specific use cases
- **Documentation**: Improve guides and tutorials
- **Testing**: Add comprehensive test coverage

## Acknowledgments

- **NASA Exoplanet Archive** for providing the comprehensive dataset
- **NASA Space Apps Challenge** for the inspiring challenge and platform
- **K2 Mission Team** for the additional exoplanet candidate data
- **Open Source Community** for the amazing tools and libraries
- **Scientific Community** for advancing exoplanet research

## Author
- Aarohi Dave
- Eshwara Pandiyan
- Yuvashree Senthilmurugan

---
