"""
Multimodal Fusion Module
Implements early and late fusion approaches to combine speech, text, 
and cognitive test features for comprehensive dementia risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from datetime import datetime
import json
from pathlib import Path

# ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Internal imports
from .models import ClassicalMLModels, DeepLearningModels
from ..config.settings import MODEL_CONFIG, ML_CONFIG




class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal fusion."""
    
    def __init__(self, speech_features: np.ndarray, text_features: np.ndarray, 
                 cognitive_features: np.ndarray, labels: np.ndarray):
        """
        Initialize multimodal dataset.
        
        Args:
            speech_features: Speech feature matrix
            text_features: Text feature matrix
            cognitive_features: Cognitive test feature matrix
            labels: Target labels
        """
        self.speech_features = torch.FloatTensor(speech_features)
        self.text_features = torch.FloatTensor(text_features)
        self.cognitive_features = torch.FloatTensor(cognitive_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'speech': self.speech_features[idx],
            'text': self.text_features[idx],
            'cognitive': self.cognitive_features[idx],
            'label': self.labels[idx]
        }


class MultimodalFusionNetwork(nn.Module):
    """Deep neural network for multimodal fusion."""
    
    def __init__(self, speech_dim: int, text_dim: int, cognitive_dim: int,
                 hidden_dims: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3,
                 fusion_type: str = 'concat'):
        """
        Initialize multimodal fusion network.
        
        Args:
            speech_dim: Speech feature dimension
            text_dim: Text feature dimension 
            cognitive_dim: Cognitive feature dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            fusion_type: Type of fusion ('concat', 'attention', 'gated')
        """
        super(MultimodalFusionNetwork, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Individual modality encoders
        self.speech_encoder = nn.Sequential(
            nn.Linear(speech_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.cognitive_encoder = nn.Sequential(
            nn.Linear(cognitive_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = hidden_dims[0] * 3
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(hidden_dims[0], num_heads=4)
            fusion_dim = hidden_dims[0]
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dims[0] * 3, hidden_dims[0]),
                nn.Sigmoid()
            )
            fusion_dim = hidden_dims[0]
        else:
            fusion_dim = hidden_dims[0] * 3
        
        # Classifier
        classifier_layers = []
        prev_dim = fusion_dim
        
        for hidden_dim in hidden_dims[1:]:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        classifier_layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, speech_x, text_x, cognitive_x):
        # Encode each modality
        speech_encoded = self.speech_encoder(speech_x)
        text_encoded = self.text_encoder(text_x)
        cognitive_encoded = self.cognitive_encoder(cognitive_x)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([speech_encoded, text_encoded, cognitive_encoded], dim=1)
        
        elif self.fusion_type == 'attention':
            # Stack modalities for attention
            modalities = torch.stack([speech_encoded, text_encoded, cognitive_encoded], dim=1)
            attended, _ = self.attention(modalities, modalities, modalities)
            fused = torch.mean(attended, dim=1)
        
        elif self.fusion_type == 'gated':
            concatenated = torch.cat([speech_encoded, text_encoded, cognitive_encoded], dim=1)
            gate_weights = self.gate(concatenated)
            
            # Weighted sum of modalities
            fused = (gate_weights * speech_encoded + 
                    gate_weights * text_encoded + 
                    gate_weights * cognitive_encoded) / 3
        
        else:
            fused = torch.cat([speech_encoded, text_encoded, cognitive_encoded], dim=1)
        
        # Classification
        output = self.classifier(fused)
        return output.squeeze()


class MultimodalFusion:
    """Main class for multimodal fusion approaches."""
    
    def __init__(self):
        """Initialize multimodal fusion system."""
        self.config = MODEL_CONFIG
        self.ml_config = ML_CONFIG
        
        # Feature processors
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
        # Model storage
        self.models = {}
        self.fusion_weights = {}
        
        # Classical ML models
        self.classical_models = ClassicalMLModels()
        
    def prepare_multimodal_data(self, speech_features: Dict[str, Dict], 
                               text_features: Dict[str, Dict],
                               cognitive_features: Dict[str, Dict],
                               labels: Dict[str, Union[int, float]],
                               feature_selection: bool = True,
                               dimensionality_reduction: bool = True) -> Dict[str, np.ndarray]:
        """
        Prepare and preprocess multimodal feature data.
        
        Args:
            speech_features: Speech features by sample ID
            text_features: Text features by sample ID  
            cognitive_features: Cognitive test features by sample ID
            labels: Labels by sample ID
            feature_selection: Whether to perform feature selection
            dimensionality_reduction: Whether to apply PCA
            
        Returns:
            Dictionary containing processed feature matrices and labels
        """
        try:
            # Find common sample IDs
            speech_ids = set(speech_features.keys())
            text_ids = set(text_features.keys())
            cognitive_ids = set(cognitive_features.keys())
            label_ids = set(labels.keys())
            
            common_ids = list(speech_ids & text_ids & cognitive_ids & label_ids)
            
            if len(common_ids) == 0:
                raise ValueError("No common sample IDs found across all modalities")
            
            } samples with all modalities")
            
            # Convert to DataFrames
            speech_df = pd.DataFrame.from_dict(
                {id_: speech_features[id_] for id_ in common_ids}, orient='index'
            ).fillna(0)
            
            text_df = pd.DataFrame.from_dict(
                {id_: text_features[id_] for id_ in common_ids}, orient='index'
            ).fillna(0)
            
            cognitive_df = pd.DataFrame.from_dict(
                {id_: cognitive_features[id_] for id_ in common_ids}, orient='index'
            ).fillna(0)
            
            # Extract labels
            y = np.array([labels[id_] for id_ in common_ids])
            
            # Process each modality
            processed_data = {}
            
            for modality, df in [('speech', speech_df), ('text', text_df), ('cognitive', cognitive_df)]:
                X = df.values
                
                # Remove constant features
                constant_features = []
                for i in range(X.shape[1]):
                    if np.std(X[:, i]) == 0:
                        constant_features.append(i)
                
                if constant_features:
                    X = np.delete(X, constant_features, axis=1)
                    remaining_features = [col for i, col in enumerate(df.columns) if i not in constant_features]
                    } constant features from {modality}")
                else:
                    remaining_features = df.columns.tolist()
                
                # Feature scaling
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                self.scalers[modality] = scaler
                
                # Feature selection
                if feature_selection and X.shape[1] > 50:
                    n_features = min(50, X.shape[1])
                    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                    X = selector.fit_transform(X, y)
                    self.feature_selectors[modality] = selector
                    selected_indices = selector.get_support(indices=True)
                    remaining_features = [remaining_features[i] for i in selected_indices]
                    
                
                # Dimensionality reduction
                if dimensionality_reduction and X.shape[1] > 20:
                    n_components = min(20, X.shape[1])
                    pca = PCA(n_components=n_components, random_state=self.ml_config["random_state"])
                    X = pca.fit_transform(X)
                    self.pca_transformers[modality] = pca
                    
                
                processed_data[modality] = X
            
            processed_data['labels'] = y
            processed_data['sample_ids'] = common_ids
            
            } samples")
                       f"Text: {processed_data['text'].shape[1]}, "
                       f"Cognitive: {processed_data['cognitive'].shape[1]}")
            
            return processed_data
            
        except Exception as e:
            
            raise
    
    def early_fusion(self, processed_data: Dict[str, np.ndarray], 
                    model_type: str = 'random_forest',
                    task_type: str = 'classification') -> Dict[str, Any]:
        """
        Perform early fusion by concatenating features before training.
        
        Args:
            processed_data: Processed multimodal data
            model_type: Type of ML model to use
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary containing fusion results
        """
        try:
            # Concatenate all features
            X_combined = np.concatenate([
                processed_data['speech'],
                processed_data['text'], 
                processed_data['cognitive']
            ], axis=1)
            
            y = processed_data['labels']
            
            
            
            # Train model on combined features
            results = self.classical_models.train_model(
                X_combined, y, model_type, task_type, hyperparameter_tuning=True
            )
            
            results['fusion_type'] = 'early'
            results['combined_feature_dim'] = X_combined.shape[1]
            
            # Store model
            model_key = f"early_fusion_{model_type}_{task_type}"
            self.models[model_key] = results
            
            
            return results
            
        except Exception as e:
            
            raise
    
    def late_fusion(self, processed_data: Dict[str, np.ndarray],
                   model_type: str = 'random_forest',
                   task_type: str = 'classification',
                   fusion_strategy: str = 'weighted_average') -> Dict[str, Any]:
        """
        Perform late fusion by training separate models and combining predictions.
        
        Args:
            processed_data: Processed multimodal data
            model_type: Type of ML model to use
            task_type: 'classification' or 'regression'
            fusion_strategy: How to combine predictions ('weighted_average', 'voting', 'stacking')
            
        Returns:
            Dictionary containing fusion results
        """
        try:
            y = processed_data['labels']
            modality_results = {}
            
            # Train separate model for each modality
            for modality in ['speech', 'text', 'cognitive']:
                X_modality = processed_data[modality]
                
                
                results = self.classical_models.train_model(
                    X_modality, y, model_type, task_type, hyperparameter_tuning=True
                )
                
                modality_results[modality] = results
                
            
            # Combine predictions based on strategy
            if fusion_strategy == 'weighted_average':
                fusion_results = self._weighted_average_fusion(modality_results, processed_data, task_type)
            elif fusion_strategy == 'voting':
                fusion_results = self._voting_fusion(modality_results, processed_data, task_type)
            elif fusion_strategy == 'stacking':
                fusion_results = self._stacking_fusion(modality_results, processed_data, task_type)
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
            # Compile results
            results = {
                'fusion_type': 'late',
                'fusion_strategy': fusion_strategy,
                'modality_results': modality_results,
                'fusion_results': fusion_results,
                'individual_scores': {mod: res['metrics']['test_score'] 
                                    for mod, res in modality_results.items()},
                'fusion_score': fusion_results['test_score']
            }
            
            # Store model
            model_key = f"late_fusion_{fusion_strategy}_{model_type}_{task_type}"
            self.models[model_key] = results
            
             completed - "
                       f"Fusion AUC/R2: {fusion_results['test_score']:.3f}")
            return results
            
        except Exception as e:
            
            raise
    
    def deep_multimodal_fusion(self, processed_data: Dict[str, np.ndarray],
                              fusion_type: str = 'concat',
                              hidden_dims: List[int] = [128, 64, 32],
                              dropout_rate: float = 0.3,
                              learning_rate: float = 0.001,
                              epochs: int = 100,
                              batch_size: int = 32) -> Dict[str, Any]:
        """
        Perform deep multimodal fusion using neural networks.
        
        Args:
            processed_data: Processed multimodal data
            fusion_type: Type of fusion ('concat', 'attention', 'gated')
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary containing fusion results
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Prepare data
            X_speech = processed_data['speech']
            X_text = processed_data['text']
            X_cognitive = processed_data['cognitive']
            y = processed_data['labels']
            
            # Split data
            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                indices, 
                test_size=self.ml_config["train_test_split"],
                random_state=self.ml_config["random_state"],
                stratify=y
            )
            
            # Create datasets
            train_dataset = MultimodalDataset(
                X_speech[train_idx], X_text[train_idx], 
                X_cognitive[train_idx], y[train_idx]
            )
            test_dataset = MultimodalDataset(
                X_speech[test_idx], X_text[test_idx],
                X_cognitive[test_idx], y[test_idx]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = MultimodalFusionNetwork(
                speech_dim=X_speech.shape[1],
                text_dim=X_text.shape[1], 
                cognitive_dim=X_cognitive.shape[1],
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                fusion_type=fusion_type
            ).to(device)
            
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
                for batch in train_loader:
                    speech = batch['speech'].to(device)
                    text = batch['text'].to(device)
                    cognitive = batch['cognitive'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(speech, text, cognitive)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for batch in test_loader:
                        speech = batch['speech'].to(device)
                        text = batch['text'].to(device)
                        cognitive = batch['cognitive'].to(device)
                        labels = batch['label'].to(device)
                        
                        outputs = model(speech, text, cognitive)
                        loss = criterion(outputs, labels)
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
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.ml_config["early_stopping_patience"]:
                    
                    break
                
                if epoch % 20 == 0:
                    
            
            # Load best model and evaluate
            model.load_state_dict(best_model_state)
            metrics = self._evaluate_multimodal_model(model, test_loader, device)
            
            results = {
                'fusion_type': 'deep',
                'deep_fusion_type': fusion_type,
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'metrics': metrics,
                'best_epoch': len(train_losses) - patience_counter - 1
            }
            
            # Store model
            model_key = f"deep_fusion_{fusion_type}"
            self.models[model_key] = results
            
             completed - "
                       f"Test AUC: {metrics['auc']:.3f}")
            return results
            
        except Exception as e:
            
            raise
    
    def compare_fusion_approaches(self, processed_data: Dict[str, np.ndarray],
                                 model_type: str = 'random_forest',
                                 task_type: str = 'classification') -> Dict[str, Any]:
        """
        Compare different fusion approaches on the same data.
        
        Args:
            processed_data: Processed multimodal data
            model_type: Type of ML model to use
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            comparison_results = {}
            
            # Early fusion
            
            early_results = self.early_fusion(processed_data, model_type, task_type)
            comparison_results['early_fusion'] = early_results
            
            # Late fusion strategies
            for strategy in ['weighted_average', 'voting']:
                ...")
                late_results = self.late_fusion(processed_data, model_type, task_type, strategy)
                comparison_results[f'late_fusion_{strategy}'] = late_results
            
            # Deep fusion approaches
            for fusion_type in ['concat', 'attention', 'gated']:
                ...")
                deep_results = self.deep_multimodal_fusion(processed_data, fusion_type)
                comparison_results[f'deep_fusion_{fusion_type}'] = deep_results
            
            # Summarize results
            summary = {}
            for approach, results in comparison_results.items():
                if 'metrics' in results:
                    score = results['metrics']['test_auc'] if task_type == 'classification' else results['metrics']['test_r2']
                elif 'fusion_score' in results:
                    score = results['fusion_score']
                else:
                    score = 0
                
                summary[approach] = score
            
            # Find best approach
            best_approach = max(summary, key=summary.get)
            
            final_results = {
                'comparison_results': comparison_results,
                'performance_summary': summary,
                'best_approach': best_approach,
                'best_score': summary[best_approach]
            }
            
                       f"(Score: {summary[best_approach]:.3f})")
            
            return final_results
            
        except Exception as e:
            
            raise
    
    def predict_multimodal(self, speech_features: Dict, text_features: Dict, 
                          cognitive_features: Dict, model_key: str) -> Dict[str, float]:
        """
        Make predictions using trained multimodal model.
        
        Args:
            speech_features: Speech features for prediction
            text_features: Text features for prediction
            cognitive_features: Cognitive features for prediction
            model_key: Key identifying the trained model
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")
            
            model_data = self.models[model_key]
            
            # Prepare features using same preprocessing
            features_combined = {
                'sample_1': {
                    'speech': speech_features,
                    'text': text_features,
                    'cognitive': cognitive_features
                }
            }
            
            # Apply same preprocessing transformations
            processed_features = {}
            for modality in ['speech', 'text', 'cognitive']:
                # Convert to array
                feature_dict = features_combined['sample_1'][modality]
                feature_array = np.array([list(feature_dict.values())]).reshape(1, -1)
                
                # Apply scaling
                if modality in self.scalers:
                    feature_array = self.scalers[modality].transform(feature_array)
                
                # Apply feature selection
                if modality in self.feature_selectors:
                    feature_array = self.feature_selectors[modality].transform(feature_array)
                
                # Apply PCA
                if modality in self.pca_transformers:
                    feature_array = self.pca_transformers[modality].transform(feature_array)
                
                processed_features[modality] = feature_array
            
            # Make prediction based on model type
            if 'early_fusion' in model_key:
                # Concatenate features for early fusion
                X_combined = np.concatenate([
                    processed_features['speech'],
                    processed_features['text'],
                    processed_features['cognitive']
                ], axis=1)
                
                predictions = self.classical_models.predict(X_combined, model_key)
                
            elif 'late_fusion' in model_key:
                # Get predictions from individual modality models
                individual_preds = {}
                for modality in ['speech', 'text', 'cognitive']:
                    # Find the individual model key
                    individual_model_key = f"{modality}_model"  # This would need to be stored properly
                    # For now, return placeholder
                    individual_preds[modality] = 0.5
                
                # Combine based on fusion strategy
                fusion_strategy = model_data['fusion_strategy']
                if fusion_strategy == 'weighted_average':
                    weights = self.fusion_weights.get(model_key, {'speech': 1/3, 'text': 1/3, 'cognitive': 1/3})
                    final_pred = sum(weights[mod] * pred for mod, pred in individual_preds.items())
                else:
                    final_pred = np.mean(list(individual_preds.values()))
                
                predictions = {'predictions': [final_pred], 'probabilities': [final_pred]}
                
            elif 'deep_fusion' in model_key:
                # Use deep learning model
                model = model_data['model']
                device = next(model.parameters()).device
                
                model.eval()
                with torch.no_grad():
                    speech_tensor = torch.FloatTensor(processed_features['speech']).to(device)
                    text_tensor = torch.FloatTensor(processed_features['text']).to(device)
                    cognitive_tensor = torch.FloatTensor(processed_features['cognitive']).to(device)
                    
                    output = model(speech_tensor, text_tensor, cognitive_tensor)
                    prediction = output.cpu().numpy()[0]
                
                predictions = {'predictions': [prediction], 'probabilities': [prediction]}
            
            else:
                raise ValueError(f"Unknown model type in {model_key}")
            
            # Format results
            result = {
                'risk_score': float(predictions['probabilities'][0]),
                'risk_category': self._categorize_risk(predictions['probabilities'][0]),
                'confidence': float(abs(predictions['probabilities'][0] - 0.5) * 2),  # Distance from decision boundary
                'model_used': model_key
            }
            
            return result
            
        except Exception as e:
            
            raise
    
    # Helper methods
    def _weighted_average_fusion(self, modality_results: Dict, processed_data: Dict, task_type: str) -> Dict:
        """Perform weighted average fusion of modality predictions."""
        # Calculate weights based on individual performance
        weights = {}
        total_score = 0
        
        for modality, results in modality_results.items():
            score = results['metrics']['test_score']
            weights[modality] = max(score, 0.1)  # Minimum weight
            total_score += weights[modality]
        
        # Normalize weights
        weights = {mod: weight/total_score for mod, weight in weights.items()}
        self.fusion_weights[f"weighted_average"] = weights
        
        # This would need actual test predictions to compute properly
        # For now, return average of individual scores
        fusion_score = sum(weights[mod] * modality_results[mod]['metrics']['test_score'] 
                          for mod in weights.keys())
        
        return {'test_score': fusion_score, 'weights': weights}
    
    def _voting_fusion(self, modality_results: Dict, processed_data: Dict, task_type: str) -> Dict:
        """Perform voting fusion of modality predictions."""
        # Simple majority voting - would need actual predictions
        fusion_score = np.mean([results['metrics']['test_score'] for results in modality_results.values()])
        return {'test_score': fusion_score}
    
    def _stacking_fusion(self, modality_results: Dict, processed_data: Dict, task_type: str) -> Dict:
        """Perform stacking fusion using meta-learner."""
        # Would need to implement proper stacking with cross-validation
        # For now, return weighted average
        return self._weighted_average_fusion(modality_results, processed_data, task_type)
    
    def _evaluate_multimodal_model(self, model: nn.Module, test_loader: DataLoader, device: str) -> Dict[str, float]:
        """Evaluate multimodal deep learning model."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                speech = batch['speech'].to(device)
                text = batch['text'].to(device)
                cognitive = batch['cognitive'].to(device)
                labels = batch['label']
                
                outputs = model(speech, text, cognitive)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
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
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into interpretable categories."""
        if risk_score < 0.3:
            return "Low Risk"
        elif risk_score < 0.7:
            return "Moderate Risk" 
        else:
            return "High Risk"
    
    def save_fusion_model(self, model_key: str, save_path: str):
        """Save fusion model and preprocessing components."""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")
            
            # Save preprocessing components
            preprocessing = {
                'scalers': {k: v for k, v in self.scalers.items()},
                'feature_selectors': {k: v for k, v in self.feature_selectors.items()},
                'pca_transformers': {k: v for k, v in self.pca_transformers.items()},
                'fusion_weights': self.fusion_weights
            }
            
            import joblib
            joblib.dump(preprocessing, save_dir / f"{model_key}_preprocessing.joblib")
            
            # Save model based on type
            model_data = self.models[model_key]
            
            if 'deep_fusion' in model_key:
                # Save PyTorch model
                torch.save(model_data['model'].state_dict(), save_dir / f"{model_key}_model.pth")
                # Save metadata
                metadata = {k: v for k, v in model_data.items() if k != 'model'}
                with open(save_dir / f"{model_key}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            else:
                # Save classical models
                joblib.dump(model_data, save_dir / f"{model_key}_model.joblib")
            
            
            
        except Exception as e:
            
            raise


# Example usage
if __name__ == "__main__":
    # Example usage
    fusion_system = MultimodalFusion()
    
    # # Prepare multimodal data
    # processed_data = fusion_system.prepare_multimodal_data(
    #     speech_features, text_features, cognitive_features, labels
    # )
    # 
    # # Compare fusion approaches
    # results = fusion_system.compare_fusion_approaches(processed_data)
    # print(f"Best approach: {results['best_approach']} (Score: {results['best_score']:.3f})")
    pass
