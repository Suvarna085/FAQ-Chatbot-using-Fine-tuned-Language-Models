import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import os
import json
import math
import numpy as np
from datetime import datetime

def compute_metrics(eval_preds):
    """
    Simplified metrics computation for language modeling.
    Just return the evaluation loss - perplexity will be computed from it.
    """
    # For language modeling, the main metric is perplexity
    # The Trainer automatically computes eval_loss
    # We can compute perplexity from the loss
    return {}  # Return empty dict to let Trainer handle the metrics

class FAQTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-small", data_dir="./data"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Resize token embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("Model and tokenizer loaded successfully")
        
    def load_datasets(self, fraction=0.1):
        """Load datasets with optional fraction sampling"""
        print("Loading datasets...")

        full_train = load_from_disk(f"{self.data_dir}/train")
        full_val = load_from_disk(f"{self.data_dir}/val")

        # Use only a fraction of the data
        if 0 < fraction < 1:
            train_size = int(len(full_train) * fraction)
            val_size = int(len(full_val) * fraction)
            self.train_dataset = full_train.select(range(train_size))
            self.val_dataset = full_val.select(range(val_size))
            print(f"Using {train_size} samples for training and {val_size} for validation.")
        else:
            self.train_dataset = full_train
            self.val_dataset = full_val
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        
    def tokenize_datasets(self, max_length=512):
        """Tokenize datasets"""
        print("Tokenizing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        self.val_dataset = self.val_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        print("Tokenization completed")
        
    def setup_training_args(self, output_dir="./faq-model"):
        """Setup training arguments with best practices"""
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=3,  # Reduced from 5 for faster training
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            
            # Optimization
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            max_grad_norm=1.0,
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=50,  # Reduced for more frequent evaluation
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,
            report_to=None,  # Change to ["wandb"] if you want to use wandb
            
            # Memory optimization
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
    def train(self):
        """Train the model"""
        print("Starting training...")
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,  # Add the compute_metrics function
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "training_args": training_args.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{training_args.output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
            
        print(f"Training completed! Model saved to {training_args.output_dir}")
        
        return trainer
        
    def run_pipeline(self):
        """Run the complete training pipeline"""
        self.load_model_and_tokenizer()
        self.load_datasets()
        self.tokenize_datasets()
        trainer = self.train()
        return trainer

def main():
    """Main training script"""
    print("=" * 50)
    print("FAQ Customer Support Model Training")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists("./data/train"):
        print("Error: Preprocessed data not found. Please run data_preprocessing.py first.")
        return
    
    # Initialize trainer
    trainer = FAQTrainer(
        model_name="microsoft/DialoGPT-small",  # Small, efficient model
        data_dir="./data"
    )
    
    # Run training pipeline
    try:
        trainer.load_model_and_tokenizer()
        trainer.load_datasets(fraction=0.025)  # Train on 10% of data
        trainer.tokenize_datasets()
        trained_model = trainer.train()
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise
        
        # Print final evaluation metrics
        eval_results = trained_model.evaluate()
        print("\n📊 Final Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Compute and display perplexity from eval_loss
        if 'eval_loss' in eval_results:
            perplexity = math.exp(eval_results['eval_loss'])
            print(f"perplexity: {perplexity:.4f}")
            
        print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()