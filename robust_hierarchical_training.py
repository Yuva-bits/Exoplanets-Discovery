#!/usr/bin/env python3
"""
Hierarchical Ensemble Model Builder
Builds the complete hierarchical ensemble model system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

# Builds the complete hierarchical ensemble model system
class HierarchicalModelBuilder:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        
    def load_data_robust(self):
        """Load data with robust CSV handling"""
        print("Loading data with robust CSV handling...")
        
        try:
            # Load cumulative dataset with proper handling
            print("Loading cumulative dataset...")
            cumulative_df = pd.read_csv('cumulative_2025.10.02_16.12.45.csv', comment='#')
            print(f"Cumulative dataset: {cumulative_df.shape}")
            
            # Load K2 dataset with proper handling
            print("Loading K2 dataset...")
            k2_df = pd.read_csv('k2pandc_2025.10.02_16.12.57.csv', comment='#')
            print(f"K2 dataset: {k2_df.shape}")
            
            # Combine datasets
            print("Combining datasets...")
            combined_df = pd.concat([cumulative_df, k2_df], ignore_index=True)
            print(f"Combined dataset: {combined_df.shape}")
            
            # Check available columns
            print(f"Available columns: {list(combined_df.columns)}")
            
            # Define legitimate features (no data leakage)
            legitimate_features = [
                'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
                'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_slogg',
                'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1',
                'koi_srad_err2', 'koi_smass', 'koi_smass_err1', 'koi_smass_err2',
                'koi_sage', 'koi_sage_err1', 'koi_sage_err2'
            ]
            
            # Filter features that exist in the dataset
            available_features = [f for f in legitimate_features if f in combined_df.columns]
            print(f"Available legitimate features: {len(available_features)}")
            print(f"Features: {available_features}")
            
            # Prepare features and target
            X = combined_df[available_features].copy()
            
            # Handle target variable
            if 'koi_disposition' in combined_df.columns:
                y = combined_df['koi_disposition'].copy()
                y = (y == 'CONFIRMED').astype(int)
            else:
                print("No koi_disposition column found!")
                return None, None, None, None
            
            # Handle missing values
            print("Handling missing values...")
            X = X.fillna(X.median())
            
            print(f"Features shape: {X.shape}")
            print(f"Target distribution: {y.value_counts().to_dict()}")
            
            # CRITICAL: Proper train/test split BEFORE any scaling
            print("Performing proper train/test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Train set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Train distribution: {y_train.value_counts().to_dict()}")
            print(f"Test distribution: {y_test.value_counts().to_dict()}")
            
            # Scale ONLY training data, then transform test data
            print("Scaling data properly...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrames
            X_train_df = pd.DataFrame(X_train_scaled, columns=available_features)
            X_test_df = pd.DataFrame(X_test_scaled, columns=available_features)
            
            self.feature_columns = available_features
            
            print("Data scaling completed without leakage!")
            
            return X_train_df, X_test_df, y_train, y_test
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\nTraining XGBoost model...")
        
        # XGBoost parameters (optimized)
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 9,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test)
        y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"XGBoost trained!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   AUC: {auc:.4f}")
        
        return accuracy, auc
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        
        # Random Forest parameters (optimized)
        rf_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42
        }
        
        self.rf_model = RandomForestClassifier(**rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test)
        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Random Forest trained!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   AUC: {auc:.4f}")
        
        return accuracy, auc
    
    def create_hierarchical_nn(self):
        """Create hierarchical neural network"""
        print("\nCreating hierarchical neural network...")
        
        # Input layers for base model outputs
        xgb_input = Input(shape=(1,), name='xgb_output')
        rf_input = Input(shape=(1,), name='rf_output')
        
        # Concatenate base model outputs
        combined = Concatenate()([xgb_input, rf_input])
        
        # Hierarchical neural network (optimized architecture)
        x = Dense(128, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='final_prediction')(x)
        
        # Create model
        model = Model(inputs=[xgb_input, rf_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Hierarchical neural network created!")
        return model
    
    def train_hierarchical_nn(self, X_train, y_train, X_test, y_test):
        """Train hierarchical neural network"""
        print("\nTraining hierarchical neural network...")
        
        # Get base model predictions for hierarchical training
        xgb_train_prob = self.xgb_model.predict_proba(X_train)[:, 1]
        rf_train_prob = self.rf_model.predict_proba(X_train)[:, 1]
        xgb_test_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        rf_test_prob = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Prepare hierarchical training data
        train_input = {
            'xgb_output': xgb_train_prob.reshape(-1, 1),
            'rf_output': rf_train_prob.reshape(-1, 1)
        }
        test_input = {
            'xgb_output': xgb_test_prob.reshape(-1, 1),
            'rf_output': rf_test_prob.reshape(-1, 1)
        }
        
        # Create and train neural network
        self.nn_model = self.create_hierarchical_nn()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss')
        ]
        
        # Train model
        print("Training hierarchical neural network...")
        history = self.nn_model.fit(
            train_input, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = self.nn_model.predict(test_input, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Hierarchical neural network trained!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   AUC: {auc:.4f}")
        
        return accuracy, auc
    
    def evaluate_hierarchical_ensemble(self, X_test, y_test):
        """Evaluate complete hierarchical ensemble"""
        print("\nEvaluating hierarchical ensemble...")
        
        # Get base model predictions
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Prepare hierarchical input
        hierarchical_input = {
            'xgb_output': xgb_prob.reshape(-1, 1),
            'rf_output': rf_prob.reshape(-1, 1)
        }
        
        # Get neural network prediction
        nn_prob = self.nn_model.predict(hierarchical_input, verbose=0).flatten()
        nn_pred = (nn_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, nn_pred)
        auc = roc_auc_score(y_test, nn_prob)
        
        # Base model metrics
        xgb_pred = (xgb_prob > 0.5).astype(int)
        rf_pred = (rf_prob > 0.5).astype(int)
        
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        rf_auc = roc_auc_score(y_test, rf_prob)
        
        # Agreement analysis
        agreement = abs(xgb_prob - rf_prob) < 0.3
        agreement_rate = np.mean(agreement)
        
        # Display results
        print("\n" + "="*80)
        print("ROBUST HIERARCHICAL ENSEMBLE RESULTS")
        print("="*80)
        
        print(f"\nHIERARCHICAL ENSEMBLE:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   AUC Score: {auc:.4f}")
        print(f"   Mean Probability: {np.mean(nn_prob):.4f}")
        print(f"   Std Probability: {np.std(nn_prob):.4f}")
        
        print(f"\nBASE MODELS:")
        print(f"   XGBoost - Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%), AUC: {xgb_auc:.4f}")
        print(f"   Random Forest - Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%), AUC: {rf_auc:.4f}")
        
        print(f"\nBASE MODEL AGREEMENT:")
        print(f"   Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
        print(f"   Cases with Agreement: {np.sum(agreement)}/{len(agreement)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, nn_pred)
        print(f"\nCONFUSION MATRIX:")
        print(f"   True Negatives:  {cm[0,0]:4d}")
        print(f"   False Positives: {cm[0,1]:4d}")
        print(f"   False Negatives: {cm[1,0]:4d}")
        print(f"   True Positives:  {cm[1,1]:4d}")
        
        # Precision, Recall, F1
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nDETAILED METRICS:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Classification report
        print(f"\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, nn_pred, target_names=['Not Exoplanet', 'Exoplanet']))
        
        return {
            'hierarchical_accuracy': accuracy,
            'hierarchical_auc': auc,
            'xgb_accuracy': xgb_accuracy,
            'xgb_auc': xgb_auc,
            'rf_accuracy': rf_accuracy,
            'rf_auc': rf_auc,
            'agreement_rate': agreement_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving trained models...")
        
        try:
            # Save base models
            joblib.dump(self.xgb_model, 'robust_xgboost_model.pkl')
            joblib.dump(self.rf_model, 'robust_random_forest_model.pkl')
            joblib.dump(self.scaler, 'robust_scaler.pkl')
            
            # Save neural network
            self.nn_model.save('robust_hierarchical_ensemble.h5')
            
            # Save feature columns
            with open('robust_feature_columns.txt', 'w') as f:
                f.write('\n'.join(self.feature_columns))
            
            print("All models saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def run_robust_training(self):
        """Run complete robust training pipeline"""
        print("ROBUST HIERARCHICAL ENSEMBLE TRAINING")
        print("=" * 70)
        
        # Load data with robust handling
        X_train, X_test, y_train, y_test = self.load_data_robust()
        if X_train is None:
            return None
        
        # Train XGBoost
        xgb_accuracy, xgb_auc = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Train Random Forest
        rf_accuracy, rf_auc = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train Hierarchical Neural Network
        nn_accuracy, nn_auc = self.train_hierarchical_nn(X_train, y_train, X_test, y_test)
        
        # Evaluate complete ensemble
        results = self.evaluate_hierarchical_ensemble(X_test, y_test)
        
        # Save models
        self.save_models()
        
        print(f"\nROBUST TRAINING COMPLETED!")
        print(f"   Hierarchical Accuracy: {results['hierarchical_accuracy']:.4f}")
        print(f"   Hierarchical AUC: {results['hierarchical_auc']:.4f}")
        print(f"   XGBoost Accuracy: {results['xgb_accuracy']:.4f}")
        print(f"   Random Forest Accuracy: {results['rf_accuracy']:.4f}")
        
        return results

def main():
    """Main model building function"""
    builder = HierarchicalModelBuilder()
    results = builder.run_robust_training()
    
    if results is not None:
        print(f"\nFINAL ROBUST RESULTS:")
        print(f"   Hierarchical Accuracy: {results['hierarchical_accuracy']:.4f}")
        print(f"   Hierarchical AUC: {results['hierarchical_auc']:.4f}")
        print(f"   XGBoost Accuracy: {results['xgb_accuracy']:.4f}")
        print(f"   Random Forest Accuracy: {results['rf_accuracy']:.4f}")
        print(f"   Base Model Agreement: {results['agreement_rate']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
