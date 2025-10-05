#!/usr/bin/env python3
"""
Hierarchical Ensemble Classifier for Exoplanet Detection
Combines XGBoost and Random Forest outputs with a neural network
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Main classifier that combines XGBoost, Random Forest, and neural network
class HierarchicalEnsembleClassifier:
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.scaler = None
        self.feature_columns = []
        self.is_loaded = False
        
    def load_base_models(self):
        """Load XGBoost and Random Forest models"""
        print("Loading base models (XGBoost + Random Forest)...")
        
        try:
            # Load XGBoost model
            if XGBOOST_AVAILABLE:
                self.xgb_model = joblib.load('robust_xgboost_model.pkl')
                print("XGBoost model loaded")
            else:
                print("XGBoost not available")
            
            # Load Random Forest model
            self.rf_model = joblib.load('robust_random_forest_model.pkl')
            print("Random Forest model loaded")
            
            # Load scaler and feature columns
            self.scaler = joblib.load('robust_scaler.pkl')
            with open('robust_feature_columns.txt', 'r') as f:
                self.feature_columns = f.read().strip().split('\n')
            
            print("Base models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading base models: {str(e)}")
            return False
    
    def create_hierarchical_nn(self):
        """Create Wide Feedforward Neural Network that takes XGBoost and RF outputs as input"""
        print("Creating Hierarchical Neural Network...")
        
        # Input layer for XGBoost and Random Forest outputs
        xgb_input = Input(shape=(1,), name='xgb_output')
        rf_input = Input(shape=(1,), name='rf_output')
        
        # Concatenate the outputs
        combined = Concatenate()([xgb_input, rf_input])
        
        # Wide Feedforward Network
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid', name='final_prediction')(x)
        
        # Create model
        model = Model(inputs=[xgb_input, rf_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Hierarchical Neural Network created!")
        return model
    
    def prepare_hierarchical_data(self, X, y):
        """Prepare data for the hierarchical model"""
        print("Preparing hierarchical data...")
        
        # Get XGBoost and Random Forest predictions
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        
        # Create hierarchical input data
        hierarchical_input = {
            'xgb_output': xgb_prob.reshape(-1, 1),
            'rf_output': rf_prob.reshape(-1, 1)
        }
        
        print(f"Hierarchical data prepared: {len(xgb_prob)} samples")
        return hierarchical_input, y
    
    def train_hierarchical_model(self, X_train, y_train, X_test, y_test):
        """Train the hierarchical neural network"""
        print("Training hierarchical neural network...")
        
        # Prepare hierarchical data
        train_input, train_labels = self.prepare_hierarchical_data(X_train, y_train)
        test_input, test_labels = self.prepare_hierarchical_data(X_test, y_test)
        
        # Create model
        model = self.create_hierarchical_nn()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_loss'),
            ModelCheckpoint('hierarchical_ensemble_best.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        print("Training neural network...")
        history = model.fit(
            train_input, train_labels,
            epochs=100,
            batch_size=32,
            validation_data=(test_input, test_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_proba = model.predict(test_input, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(test_labels, y_pred)
        auc_score = roc_auc_score(test_labels, y_pred_proba)
        
        print(f"Hierarchical Neural Network Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        
        # Save model
        model.save('hierarchical_ensemble.h5')
        print("Hierarchical model saved!")
        
        self.nn_model = model
        return model, y_pred, y_pred_proba.flatten()
    
    def load_hierarchical_model(self):
        """Load the trained hierarchical model"""
        print("Loading hierarchical model...")
        
        try:
            self.nn_model = tf.keras.models.load_model('robust_hierarchical_ensemble.h5')
            print("Hierarchical model loaded!")
            return True
        except Exception as e:
            print(f"Error loading hierarchical model: {str(e)}")
            return False
    
    def predict_hierarchical(self, X):
        """Make predictions using hierarchical ensemble"""
        if not self.is_loaded:
            raise ValueError("Models not loaded. Please call load_models() first.")
        
        # Get base model predictions
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        
        # Prepare hierarchical input
        hierarchical_input = {
            'xgb_output': xgb_prob.reshape(-1, 1),
            'rf_output': rf_prob.reshape(-1, 1)
        }
        
        # Get neural network prediction
        nn_prob = self.nn_model.predict(hierarchical_input, verbose=0).flatten()
        nn_pred = (nn_prob > 0.5).astype(int)
        
        return nn_pred, nn_prob, xgb_prob, rf_prob
    
    def load_models(self):
        """Load all models for hierarchical ensemble"""
        print("Loading hierarchical ensemble models...")
        
        if not self.load_base_models():
            return False
        
        if not self.load_hierarchical_model():
            return False
        
        self.is_loaded = True
        print("All hierarchical models loaded successfully!")
        return True
    
    def classify_exoplanet(self, data):
        """Classify whether a celestial body is an exoplanet using hierarchical ensemble"""
        if not self.is_loaded:
            raise ValueError("Models not loaded. Please call load_models() first.")
        
        # Prepare input data
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be dict or DataFrame")
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[self.feature_columns]
        
        # Scale the data
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_columns)
        
        # Make hierarchical prediction
        nn_pred, nn_prob, xgb_prob, rf_prob = self.predict_hierarchical(df_scaled)
        
        result = {
            'is_exoplanet': bool(nn_pred[0]),
            'confidence': float(nn_prob[0]),
            'hierarchical_prediction': nn_pred[0],
            'hierarchical_probability': nn_prob[0],
            'xgb_probability': xgb_prob[0],
            'rf_probability': rf_prob[0],
            'base_model_agreement': abs(xgb_prob[0] - rf_prob[0]) < 0.3
        }
        
        return result
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': self.is_loaded,
            'xgboost_available': self.xgb_model is not None,
            'random_forest_available': self.rf_model is not None,
            'hierarchical_nn_available': self.nn_model is not None,
            'feature_count': len(self.feature_columns),
            'architecture': 'Hierarchical: XGBoost + RF → Wide Feedforward NN'
        }
        return info

def main():
    """Main function to test the hierarchical ensemble classifier"""
    print("Hierarchical Ensemble Classifier for Exoplanet Detection")
    print("=" * 70)
    print("Architecture: XGBoost + Random Forest → Wide Feedforward Neural Network")
    print("=" * 70)
    
    # Initialize classifier
    classifier = HierarchicalEnsembleClassifier()
    
    # Load models
    if not classifier.load_models():
        print("Failed to load models. Please ensure models are built first.")
        return
    
    # Get model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Models loaded: {info['models_loaded']}")
    print(f"  XGBoost available: {info['xgboost_available']}")
    print(f"  Random Forest available: {info['random_forest_available']}")
    print(f"  Hierarchical NN available: {info['hierarchical_nn_available']}")
    print(f"  Architecture: {info['architecture']}")
    
    # Test with sample data
    print(f"\nTesting with sample data...")
    sample_data = {
        'orbital_period': 365.0,
        'planetary_radius': 1.0,
        'transit_depth': 1000.0,
        'transit_duration': 3.0,
        'impact_parameter': 0.5,
        'stellar_temperature': 5778.0,
        'stellar_radius': 1.0,
        'stellar_gravity': 4.4,
        'equilibrium_temp': 288.0,
        'insolation': 1.0,
        'planetary_mass': 1.0,
        'semi_major_axis': 1.0,
        'orbital_eccentricity': 0.0,
        'stellar_mass': 1.0,
        'stellar_metallicity': 0.0,
        'system_distance': 10.0,
        'planetary_density': 5.5,
        'transit_depth_radius_ratio': 1000.0,
        'orbital_velocity': 2.0,
        'stellar_luminosity_proxy': 1.0
    }
    
    # Make prediction
    result = classifier.classify_exoplanet(sample_data)
    
    print(f"\nHierarchical Classification Result:")
    print(f"  Is Exoplanet: {'Yes' if result['is_exoplanet'] else 'No'}")
    print(f"  Hierarchical Confidence: {result['confidence']:.4f}")
    print(f"  XGBoost Probability: {result['xgb_probability']:.4f}")
    print(f"  Random Forest Probability: {result['rf_probability']:.4f}")
    print(f"  Base Model Agreement: {'Yes' if result['base_model_agreement'] else 'No'}")
    
    print(f"\nHierarchical Ensemble Classifier is ready for use!")

if __name__ == "__main__":
    main()
