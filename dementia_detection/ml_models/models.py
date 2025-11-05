"""
Machine Learning Models Module
Implements classical ML models (Random Forest, XGBoost) and deep learning
model interfaces for dementia risk prediction using multimodal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Classical ML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras

from ..config.settings import ML_CONFIG, MODEL_CONFIG




class DementiaDataset(Dataset):
    """PyTorch Dataset for dementia prediction."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix
            labels: Target labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepDementiaNet(nn.Module):
    """Deep neural network for dementia prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3, num_classes: int = 1):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            num_classes: Number of output classes (1 for binary classification)
        """
        super(DeepDementiaNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        if num_classes == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ClassicalMLModels:
    """Classical machine learning models for dementia prediction."""
    
    def __init__(self):
        """Initialize classical ML models."""
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        self.config = ML_CONFIG
        
        # Initialize model configurations
        self.model_configs = {
            'random_forest': {
                'classifier': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config["random_state"]
                ),
                'regressor': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config["random_state"]
                )
            },
            'xgboost': {
                'classifier': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config["random_state"]
                ),
                'regressor': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config["random_state"]
                )
            },
            'lightgbm': {
                'classifier': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config["random_state"]
                ),
                'regressor': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config["random_state"]
                )
            },
            'gradient_boosting': {
                'classifier': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.config["random_state"]
                ),
                'regressor': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.config["random_state"]
                )
            },
            'logistic_regression': {
                'classifier': LogisticRegression(
                    random_state=self.config["random_state"],
                    max_iter=1000
                ),
                'regressor': LinearRegression()
            },
            'svm': {
                'classifier': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.config["random_state"]
                ),
                'regressor': SVR(kernel='rbf')
            }
        }
    
    def prepare_data(self, features: Dict[str, Dict], labels: Dict[str, Union[int, float]], 
                    feature_selection: bool = True, n_features: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare and preprocess data for ML models.
        
        Args:
            features: Dictionary of feature dictionaries by sample ID
            labels: Dictionary of labels by sample ID  
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame.from_dict(features, orient='index')
            
            # Handle missing values
            feature_df = feature_df.fillna(0)
            
            # Align features and labels
            common_ids = list(set(feature_df.index) & set(labels.keys()))
            feature_df = feature_df.loc[common_ids]
            
            # Extract features and labels
            X = feature_df.values
            y = np.array([labels[idx] for idx in common_ids])
            feature_names = feature_df.columns.tolist()
            
            # Remove constant features
            constant_features = []
            for i, col in enumerate(feature_names):
                if np.std(X[:, i]) == 0:
                    constant_features.append(i)
            
            if constant_features:
                X = np.delete(X, constant_features, axis=1)
                feature_names = [name for i, name in enumerate(feature_names) if i not in constant_features]
            
            # Feature scaling
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers['main'] = scaler
            
            # Feature selection
            if feature_selection and len(feature_names) > n_features:
                if len(np.unique(y)) > 2:  # Regression
                    selector = SelectKBest(score_func=f_classif, k=min(n_features, len(feature_names)))
                else:  # Classification
                    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(feature_names)))
                
                X = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                feature_names = [feature_names[i] for i in selected_indices]
                self.feature_selectors['main'] = selector
            
            
            return X, y, feature_names
            
        except Exception as e:
            
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   model_type: str = 'random_forest',
                   task_type: str = 'classification',
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train a classical ML model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            task_type: 'classification' or 'regression'
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing model and training results
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config["train_test_split"],
                random_state=self.config["random_state"],
                stratify=y if task_type == 'classification' else None
            )
            
            # Get base model
            model = self.model_configs[model_type][task_type.replace('classification', 'classifier')]
            
            # Hyperparameter tuning
            if hyperparameter_tuning:
                param_grid = self._get_param_grid(model_type, task_type)
                if param_grid:
                    scoring = 'roc_auc' if task_type == 'classification' else 'r2'
                    grid_search = GridSearchCV(
                        model, param_grid,
                        cv=self.config["cv_folds"],
                        scoring=scoring,
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba_test = model.predict_proba(X_test)[:, 1]
            else:
                y_proba_train = y_pred_train
                y_proba_test = y_pred_test
            
            # Evaluate model
            metrics = self._evaluate_model(
                y_train, y_test, y_pred_train, y_pred_test,
                y_proba_train, y_proba_test, task_type
            )
            
            # Feature importance
            feature_importance = self._get_feature_importance(model)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config["cv_folds"],
                scoring='roc_auc' if task_type == 'classification' else 'r2'
            )
            
            # Store model
            model_key = f"{model_type}_{task_type}"
            self.models[model_key] = model
            
            results = {
                'model': model,
                'model_type': model_type,
                'task_type': task_type,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            
            return results
            
        except Exception as e:
            
            raise
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      model_types: List[str] = ['random_forest', 'xgboost', 'lightgbm'],
                      task_type: str = 'classification') -> Dict[str, Any]:
        """
        Train ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_types: List of model types to include in ensemble
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary containing ensemble results
        """
        try:
            ensemble_models = {}
            ensemble_predictions = {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["train_test_split"],
                random_state=self.config["random_state"],
                stratify=y if task_type == 'classification' else None
            )
            
            # Train individual models
            for model_type in model_types:
                try:
                    model = self.model_configs[model_type][task_type.replace('classification', 'classifier')]
                    model.fit(X_train, y_train)
                    
                    if task_type == 'classification' and hasattr(model, 'predict_proba'):
                        pred_test = model.predict_proba(X_test)[:, 1]
                    else:
                        pred_test = model.predict(X_test)
                    
                    ensemble_models[model_type] = model
                    ensemble_predictions[model_type] = pred_test
                    
                except Exception as e:
                    
            
            if not ensemble_models:
                raise ValueError("No models successfully trained")
            
            # Ensemble prediction (simple averaging)
            pred_arrays = list(ensemble_predictions.values())
            ensemble_pred = np.mean(pred_arrays, axis=0)
            
            # Weighted ensemble (based on individual performance)
            weights = {}
            for model_type, model in ensemble_models.items():
                if task_type == 'classification' and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_train)[:, 1]
                    score = roc_auc_score(y_train, pred)
                else:
                    pred = model.predict(X_train)
                    score = r2_score(y_train, pred) if task_type == 'regression' else accuracy_score(y_train, pred > 0.5)
                weights[model_type] = max(score, 0.1)  # Minimum weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Weighted ensemble prediction
            weighted_pred = np.zeros(len(y_test))
            for model_type, weight in weights.items():
                weighted_pred += weight * ensemble_predictions[model_type]
            
            # Evaluate ensemble
            if task_type == 'classification':
                ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
                weighted_pred_binary = (weighted_pred > 0.5).astype(int)
                
                simple_metrics = {
                    'accuracy': accuracy_score(y_test, ensemble_pred_binary),
                    'auc': roc_auc_score(y_test, ensemble_pred),
                    'precision': precision_score(y_test, ensemble_pred_binary),
                    'recall': recall_score(y_test, ensemble_pred_binary),
                    'f1': f1_score(y_test, ensemble_pred_binary)
                }
                
                weighted_metrics = {
                    'accuracy': accuracy_score(y_test, weighted_pred_binary),
                    'auc': roc_auc_score(y_test, weighted_pred),
                    'precision': precision_score(y_test, weighted_pred_binary),
                    'recall': recall_score(y_test, weighted_pred_binary),
                    'f1': f1_score(y_test, weighted_pred_binary)
                }
            else:
                simple_metrics = {
                    'mse': mean_squared_error(y_test, ensemble_pred),
                    'r2': r2_score(y_test, ensemble_pred)
                }
                
                weighted_metrics = {
                    'mse': mean_squared_error(y_test, weighted_pred),
                    'r2': r2_score(y_test, weighted_pred)
                }
            
            results = {
                'models': ensemble_models,
                'weights': weights,
                'simple_ensemble_metrics': simple_metrics,
                'weighted_ensemble_metrics': weighted_metrics,
                'predictions': {
                    'simple_ensemble': ensemble_pred,
                    'weighted_ensemble': weighted_pred,
                    'individual': ensemble_predictions
                }
            }
            
            } models")
            return results
            
        except Exception as e:
            
            raise
    
    def predict(self, X: np.ndarray, model_key: str) -> Dict[str, np.ndarray]:
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix
            model_key: Key identifying the trained model
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        try:
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")
            
            model = self.models[model_key]
            
            # Apply same preprocessing
            if 'main' in self.scalers:
                X = self.scalers['main'].transform(X)
            
            if 'main' in self.feature_selectors:
                X = self.feature_selectors['main'].transform(X)
            
            # Make predictions
            predictions = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                if probabilities.shape[1] == 2:  # Binary classification
                    probabilities = probabilities[:, 1]
            else:
                probabilities = predictions
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
        except Exception as e:
            
            raise
    
    def _get_param_grid(self, model_type: str, task_type: str) -> Dict[str, List]:
        """Get hyperparameter grid for model tuning."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        return param_grids.get(model_type, {})
    
    def _evaluate_model(self, y_train: np.ndarray, y_test: np.ndarray,
                       y_pred_train: np.ndarray, y_pred_test: np.ndarray,
                       y_proba_train: np.ndarray, y_proba_test: np.ndarray,
                       task_type: str) -> Dict[str, float]:
        """Evaluate model performance."""
        metrics = {}
        
        if task_type == 'classification':
            # Training metrics
            metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
            metrics['train_auc'] = roc_auc_score(y_train, y_proba_train)
            metrics['train_precision'] = precision_score(y_train, y_pred_train, zero_division=0)
            metrics['train_recall'] = recall_score(y_train, y_pred_train, zero_division=0)
            metrics['train_f1'] = f1_score(y_train, y_pred_train, zero_division=0)
            
            # Test metrics
            metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
            metrics['test_auc'] = roc_auc_score(y_test, y_proba_test)
            metrics['test_precision'] = precision_score(y_test, y_pred_test, zero_division=0)
            metrics['test_recall'] = recall_score(y_test, y_pred_test, zero_division=0)
            metrics['test_f1'] = f1_score(y_test, y_pred_test, zero_division=0)
            
            metrics['test_score'] = metrics['test_auc']  # Primary metric
            
        else:  # Regression
            # Training metrics
            metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
            metrics['train_r2'] = r2_score(y_train, y_pred_train)
            
            # Test metrics  
            metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
            metrics['test_r2'] = r2_score(y_test, y_pred_test)
            
            metrics['test_score'] = metrics['test_r2']  # Primary metric
        
        return metrics
    
    def _get_feature_importance(self, model) -> Optional[np.ndarray]:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            return None


class DeepLearningModels:
    """Deep learning models for dementia prediction."""
    
    def __init__(self, device: str = None):
        """Initialize deep learning models."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.config = ML_CONFIG
        
        
    
    def train_pytorch_model(self, X: np.ndarray, y: np.ndarray,
                           hidden_dims: List[int] = [256, 128, 64],
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.001,
                           epochs: int = 100,
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        Train PyTorch neural network.
        
        Args:
            X: Feature matrix
            y: Target labels
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary containing model and training results
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["train_test_split"],
                random_state=self.config["random_state"],
                stratify=y
            )
            
            # Create datasets
            train_dataset = DementiaDataset(X_train, y_train)
            test_dataset = DementiaDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = DeepDementiaNet(
                input_dim=X.shape[1],
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            train_losses = []
            test_losses = []
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for batch_features, batch_labels in test_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = model(batch_features).squeeze()
                        loss = criterion(outputs, batch_labels)
                        test_loss += loss.item()
                
                train_loss /= len(train_loader)
                test_loss /= len(test_loader)
                
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
                scheduler.step(test_loss)
                
                # Early stopping
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config["early_stopping_patience"]:
                    
                    break
                
                if epoch % 20 == 0:
                    
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate model
            metrics = self._evaluate_pytorch_model(model, test_loader)
            
            results = {
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'metrics': metrics,
                'best_epoch': len(train_losses) - patience_counter - 1
            }
            
            
            return results
            
        except Exception as e:
            
            raise
    
    def train_tensorflow_model(self, X: np.ndarray, y: np.ndarray,
                              hidden_dims: List[int] = [256, 128, 64],
                              dropout_rate: float = 0.3,
                              learning_rate: float = 0.001,
                              epochs: int = 100,
                              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train TensorFlow/Keras neural network.
        
        Args:
            X: Feature matrix
            y: Target labels
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary containing model and training results
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["train_test_split"],
                random_state=self.config["random_state"],
                stratify=y
            )
            
            # Build model
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(X.shape[1],)))
            
            for hidden_dim in hidden_dims:
                model.add(keras.layers.Dense(hidden_dim, activation='relu'))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Dropout(dropout_rate))
            
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': test_accuracy,
                'auc': test_auc,
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            results = {
                'model': model,
                'history': history.history,
                'metrics': metrics,
                'epochs_trained': len(history.history['loss'])
            }
            
            
            return results
            
        except Exception as e:
            
            raise
    
    def _evaluate_pytorch_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate PyTorch model."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features).squeeze()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        pred_binary = (all_predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(all_labels, pred_binary),
            'auc': roc_auc_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, pred_binary, zero_division=0),
            'recall': recall_score(all_labels, pred_binary, zero_division=0),
            'f1': f1_score(all_labels, pred_binary, zero_division=0)
        }


class ModelManager:
    """Manage and coordinate different ML models."""
    
    def __init__(self):
        """Initialize model manager."""
        self.classical_models = ClassicalMLModels()
        self.deep_models = DeepLearningModels()
        self.trained_models = {}
        
    def save_model(self, model_key: str, model_data: Dict, save_path: str):
        """Save trained model to disk."""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save different types of models differently
            if 'pytorch' in model_key.lower():
                # Save PyTorch model
                torch.save(model_data['model'].state_dict(), save_dir / f"{model_key}_model.pth")
                # Save metadata
                metadata = {k: v for k, v in model_data.items() if k != 'model'}
                with open(save_dir / f"{model_key}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            elif 'tensorflow' in model_key.lower():
                # Save TensorFlow model
                model_data['model'].save(save_dir / f"{model_key}_model")
                # Save metadata
                metadata = {k: v for k, v in model_data.items() if k != 'model'}
                with open(save_dir / f"{model_key}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            else:
                # Save classical ML model
                joblib.dump(model_data['model'], save_dir / f"{model_key}_model.joblib")
                # Save metadata
                metadata = {k: v for k, v in model_data.items() if k != 'model'}
                with open(save_dir / f"{model_key}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            
            
        except Exception as e:
            
            raise
    
    def load_model(self, model_key: str, load_path: str) -> Dict:
        """Load trained model from disk."""
        try:
            load_dir = Path(load_path)
            
            # Load metadata
            with open(load_dir / f"{model_key}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load model based on type
            if 'pytorch' in model_key.lower():
                # Would need to reconstruct model architecture
                
                return metadata
            
            elif 'tensorflow' in model_key.lower():
                model = keras.models.load_model(load_dir / f"{model_key}_model")
                metadata['model'] = model
                return metadata
            
            else:
                model = joblib.load(load_dir / f"{model_key}_model.joblib")
                metadata['model'] = model
                return metadata
            
        except Exception as e:
            
            raise


# Example usage
if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    
    # # Train classical models
    # features = {...}  # Dictionary of features by sample ID
    # labels = {...}    # Dictionary of labels by sample ID
    # 
    # X, y, feature_names = manager.classical_models.prepare_data(features, labels)
    # results = manager.classical_models.train_model(X, y, 'random_forest', 'classification')
    # print(f"Random Forest AUC: {results['metrics']['test_auc']:.3f}")
    pass
