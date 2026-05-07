"""
BERT-based Classifier for biomedical abstract classification.
Uses pre-trained BERT model fine-tuned for medical specialty classification.
"""

import logging
import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Suppress FutureWarning from Hugging Face
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')

# Set Hugging Face cache directory and token settings
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningClassifier:
    """
    BERT-based classifier for biomedical abstract classification.
    Fine-tunes a pre-trained BERT model for medical specialty classification.
    """
    
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
    
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 max_len: int = 512,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Hugging Face model identifier (default: distilbert-base-uncased for speed)
            max_len: Maximum sequence length (512 for BERT)
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            num_epochs: Number of training epochs
        """
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info(f"Loading tokenizer: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"✓ Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        self.model = None
        
        self.class_names = list(self.SPECIALTY_KEYWORDS.keys()) + ['Other']
        self.label_encoder = {label: idx for idx, label in enumerate(self.class_names)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
        logger.info(f"Initialized BERT Classifier (Model: {model_name}, MaxLen: {max_len})")
    
    def auto_label(self, abstracts: List[str]) -> List[str]:
        """Auto-label abstracts based on keywords. Handles NaN values gracefully."""
        labels = []
        
        for abstract in abstracts:
            # Handle NaN/None values
            if not isinstance(abstract, str) or len(str(abstract).strip()) == 0:
                labels.append('Other')
                continue
            
            abstract_lower = abstract.lower()
            best_label = 'Other'
            max_matches = 0
            
            for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in abstract_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_label = specialty
            
            labels.append(best_label)
        
        return labels
    
    def prepare_data(self, abstracts: List[str], labels: List[str] = None) -> Tuple:
        """
        Prepare data for BERT training.
        
        Args:
            abstracts: List of abstract texts
            labels: List of labels (if None, auto-label)
            
        Returns:
            Tuple of (input_ids, attention_masks, labels_encoded)
        """
        # Clean abstracts
        cleaned_abstracts = []
        invalid_count = 0
        empty_count = 0
        
        for abstract in abstracts:
            if isinstance(abstract, str) and len(abstract.strip()) > 0:
                cleaned_abstracts.append(abstract)
            elif not isinstance(abstract, str):
                invalid_count += 1
                continue
            else:
                empty_count += 1
                continue
        
        total_removed = invalid_count + empty_count
        if total_removed > 0:
            pct = (total_removed / len(abstracts)) * 100 if len(abstracts) > 0 else 0
            logger.info(f"Data cleaning summary: Removed {total_removed} items ({pct:.1f}%) - "
                       f"{invalid_count} non-string values (NaN/float), {empty_count} empty strings. "
                       f"Kept {len(cleaned_abstracts)} valid abstracts.")
        
        logger.info(f"Prepared {len(cleaned_abstracts)} abstracts for training")
        
        if len(cleaned_abstracts) == 0:
            raise ValueError("No valid abstracts found after cleaning")
        
        abstracts = cleaned_abstracts
        
        # Auto-label if not provided
        if labels is None:
            labels = self.auto_label(abstracts)
        
        labels = labels[:len(abstracts)]
        
        # Encode labels
        y = np.array([self.label_encoder[label] for label in labels])
        
        # Tokenize with BERT tokenizer
        logger.info("Tokenizing abstracts with BERT...")
        encoded = self.tokenizer(
            abstracts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        attention_masks = encoded['attention_mask']
        labels_tensor = torch.tensor(y, dtype=torch.long)
        
        logger.info(f"Tokenized {len(abstracts)} abstracts")
        logger.info(f"Input shape: {input_ids.shape}")
        
        return input_ids, attention_masks, labels_tensor
    
    def train(self, 
              abstracts: List[str], 
              labels: List[str] = None,
              test_size: float = 0.2,
              epochs: int = None,
              verbose: int = 1) -> Dict:
        """
        Fine-tune BERT model for classification.
        
        Args:
            abstracts: List of abstract texts
            labels: List of labels (if None, auto-label)
            test_size: Proportion for testing
            epochs: Number of training epochs (uses self.num_epochs if None)
            verbose: Verbosity level
            
        Returns:
            Dictionary with training metrics
        """
        if epochs is None:
            epochs = self.num_epochs
        
        logger.info(f"Training BERT classifier on {len(abstracts)} abstracts...")
        
        # Prepare data
        input_ids, attention_masks, y = self.prepare_data(abstracts, labels)
        
        # Split data
        X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
            input_ids, attention_masks, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train_ids)}, Test set: {len(X_test_ids)}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_ids, X_train_masks, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_ids, X_test_masks, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Initialize BERT model
        logger.info(f"Loading pre-trained {self.model_name}...")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.class_names),
                output_attentions=False,
                output_hidden_states=False,
                trust_remote_code=True
            )
            logger.info(f"✓ Model loaded successfully")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("Starting model training...")
        self.model.train()
        
        for epoch in range(epochs):
            total_train_loss = 0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                input_ids_batch = batch[0].to(self.device)
                attention_mask_batch = batch[1].to(self.device)
                labels_batch = batch[2].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    labels=labels_batch
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Avg Training Loss: {avg_train_loss:.4f}")
        
        # Evaluation on test set
        logger.info("Evaluating on test set...")
        self.model.eval()
        
        y_pred_all = []
        y_test_all = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids_batch = batch[0].to(self.device)
                attention_mask_batch = batch[1].to(self.device)
                labels_batch = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch
                )
                
                logits = outputs.logits
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred_all.extend(y_pred)
                y_test_all.extend(labels_batch.cpu().numpy())
        
        y_test_all = np.array(y_test_all)
        y_pred_all = np.array(y_pred_all)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test_all, y_pred_all)
        test_f1 = f1_score(y_test_all, y_pred_all, average='weighted', zero_division=0)
        
        metrics = {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'confusion_matrix': confusion_matrix(y_test_all, y_pred_all),
            'classification_report': classification_report(
                y_test_all, y_pred_all,
                target_names=self.class_names,
                zero_division=0
            ),
            'test_labels': y_test_all,
            'test_predictions': y_pred_all
        }
        
        logger.info(f"Training complete. Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        return metrics
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict label for a single abstract.
        
        Args:
            text: Abstract text
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle NaN/None values
        if not isinstance(text, str):
            text = "other"
        
        # Truncate to max_len - 2 to account for [CLS] and [SEP] tokens
        text = text[:self.max_len * 4]  # Rough estimate
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx]
        
        return self.label_decoder[pred_idx], float(confidence)
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict labels for multiple abstracts."""
        results = []
        for text in texts:
            label, confidence = self.predict(text)
            results.append((label, confidence))
        return results
    
    def save_model(self, output_dir: Path):
        """Save trained model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(output_dir))
        logger.info(f"Model saved to {output_dir}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Tokenizer saved to {output_dir}")
    
    def load_model(self, model_dir: Path):
        """Load trained model and tokenizer."""
        model_dir = Path(model_dir)
        
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        logger.info(f"Model and tokenizer loaded from {model_dir}")
