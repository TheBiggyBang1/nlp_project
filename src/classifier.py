"""
Machine Learning Classification module for biomedical abstracts.
Auto-labels abstracts and trains classifier for medical specialties.
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class MedicalClassifier:
    """
    Auto-label and classify biomedical abstracts into medical specialties.
    """
    
    # Keywords for each medical specialty
    SPECIALTY_KEYWORDS = {
        'Oncology': {
            'cancer', 'tumor', 'carcinoma', 'melanoma', 'leukemia', 
            'lymphoma', 'sarcoma', 'metastasis', 'neoplasm', 'malignant',
            'chemotherapy', 'radiotherapy', 'oncogenic', 'oncology'
        },
        'Cardiology': {
            'heart', 'cardiac', 'coronary', 'hypertension', 'arrhythmia',
            'myocardial', 'infarction', 'atherosclerosis', 'heart disease',
            'ventricular', 'angina', 'cardiology', 'cardiovascular'
        },
        'Infectious Disease': {
            'infection', 'virus', 'bacterial', 'antibacterial', 'vaccine',
            'covid', 'pandemic', 'pathogen', 'infectious', 'antibiotics',
            'antibiotic', 'immunodeficiency', 'fever', 'sepsis', 'flu'
        },
        'Neurology': {
            'brain', 'neurological', 'parkinson', 'alzheimer', 'epilepsy',
            'stroke', 'neurodegeneration', 'neural', 'neurotransmitter',
            'synapse', 'neuroinflammation', 'neurology', 'dementia'
        }
    }
    
    def __init__(self, classifier_type: str = 'rf'):
        """
        Initialize classifier.
        
        Args:
            classifier_type: 'rf' for RandomForest or 'svm' for SVM
        """
        self.classifier_type = classifier_type
        self.tfidf = None
        self.classifier = None
        self.label_encoder = None
        self.class_names = list(self.SPECIALTY_KEYWORDS.keys()) + ['Other']
        
        logger.info(f"Initialized classifier with type: {classifier_type}")
    
    def auto_label(self, abstracts: List[str]) -> List[str]:
        """
        Automatically label abstracts based on keyword matching.
        
        Args:
            abstracts: List of abstract texts
            
        Returns:
            List of labels
        """
        labels = []
        
        for abstract in abstracts:
            abstract_lower = abstract.lower()
            
            # Find best matching specialty
            best_label = 'Other'
            max_matches = 0
            
            for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in abstract_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_label = specialty
            
            labels.append(best_label)
        
        return labels
    
    def get_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """
        Get distribution of classes.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary with class counts
        """
        from collections import Counter
        return dict(Counter(labels))
    
    def train(self, abstracts: List[str], labels: List[str] = None,
              test_size: float = 0.2, max_features: int = 500,
              ngram_range: Tuple[int, int] = (1, 2)):
        """
        Train classifier on abstracts.
        
        Args:
            abstracts: List of abstract texts
            labels: List of labels (if None, auto-label)
            test_size: Proportion of data for testing
            max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        logger.info(f"Training {self.classifier_type} classifier...")
        
        # Auto-label if not provided
        if labels is None:
            labels = self.auto_label(abstracts)
        
        # Encode labels
        self.label_encoder = {label: idx for idx, label in enumerate(self.class_names)}
        y = np.array([self.label_encoder[label] for label in labels])
        
        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        X = self.tfidf.fit_transform(abstracts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train classifier
        if self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:  # SVM
            self.classifier = LinearSVC(
                random_state=42,
                max_iter=2000,
                class_weight='balanced'
            )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.classifier.predict(X_train)
        test_pred = self.classifier.predict(X_test)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_f1_macro': f1_score(y_test, test_pred, average='macro', zero_division=0),
            'test_f1_weighted': f1_score(y_test, test_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(
                y_test, test_pred, 
                target_names=self.class_names,
                zero_division=0
            ),
            'test_labels': y_test,
            'test_predictions': test_pred
        }
        
        logger.info(f"Model trained. Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict label for a single abstract.
        
        Args:
            text: Abstract text
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.classifier is None:
            raise ValueError("Model not trained")
        
        # Vectorize
        X = self.tfidf.transform([text])
        
        # Predict
        pred_idx = self.classifier.predict(X)[0]
        label = self.class_names[pred_idx]
        
        # Get confidence (probability for RF, decision function for SVM)
        if self.classifier_type == 'rf':
            confidence = max(self.classifier.predict_proba(X)[0])
        else:
            # Use decision function scores for SVM
            scores = self.classifier.decision_function(X)[0]
            confidence = 1.0 / (1.0 + np.exp(-scores[pred_idx]))  # Sigmoid approximation
        
        return label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict labels for multiple abstracts.
        
        Args:
            texts: List of abstract texts
            
        Returns:
            List of (label, confidence) tuples
        """
        if self.classifier is None:
            raise ValueError("Model not trained")
        
        X = self.tfidf.transform(texts)
        predictions = self.classifier.predict(X)
        
        results = []
        if self.classifier_type == 'rf':
            probabilities = self.classifier.predict_proba(X)
            for idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
                label = self.class_names[pred]
                confidence = max(probs)
                results.append((label, confidence))
        else:
            scores = self.classifier.decision_function(X)
            for idx, pred in enumerate(predictions):
                label = self.class_names[pred]
                confidence = 1.0 / (1.0 + np.exp(-scores[idx, pred]))
                results.append((label, confidence))
        
        return results
    
    def save_model(self, model_dir: Path):
        """
        Save trained classifier and TF-IDF vectorizer.
        
        Args:
            model_dir: Directory to save models
        """
        if self.classifier is None:
            raise ValueError("No model to save")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(model_dir / 'tfidf.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)
        
        logger.info(f"Saved model to {model_dir}")
    
    def load_model(self, model_dir: Path):
        """
        Load trained classifier and TF-IDF vectorizer.
        
        Args:
            model_dir: Directory containing saved models
        """
        with open(model_dir / 'classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(model_dir / 'tfidf.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)
        
        logger.info(f"Loaded model from {model_dir}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             output_path: Path = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_class_distribution(self, labels: List[str], output_path: Path = None):
        """
        Plot class distribution.
        
        Args:
            labels: List of labels
            output_path: Path to save figure
        """
        dist = self.get_class_distribution(labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(dist.keys(), dist.values(), color='coral', edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
