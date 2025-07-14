import numpy as np
import torch
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
import os
import json
import math
from datetime import datetime

def compute_perplexity(eval_preds):
    """
    Compute perplexity for language modeling.
    For generative models, perplexity is more meaningful than accuracy.
    """
    logits, labels = eval_preds
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Only compute loss on valid tokens (not -100)
    valid_indices = shift_labels != -100
    
    if valid_indices.sum() == 0:
        return {"perplexity": float('inf')}
    
    # Get predictions and filter valid tokens
    valid_logits = shift_logits[valid_indices]
    valid_labels = shift_labels[valid_indices]
    
    # Compute cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(valid_logits, valid_labels)
    
    # Compute perplexity
    perplexity = torch.exp(loss).item()
    
    # Also compute token-level accuracy
    predictions = torch.argmax(valid_logits, dim=-1)
    token_accuracy = (predictions == valid_labels).float().mean().item()
    
    return {
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "eval_loss": loss.item()
    }

def evaluate_model_simple(model, tokenizer, test_dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Simple evaluation function that computes perplexity and loss.
    """
    model.eval()
    model.to(device)
    
    # Fix: Properly tokenize the dataset
    print("📝 Tokenizing test dataset...")
    
    def tokenize_function(examples):
        # Handle both single strings and lists of strings
        texts = examples["text"]
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # Don't pad here, let the data collator handle it
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
        
        return tokenized
    
    # Tokenize the dataset
    tokenized_dataset = test_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"✅ Tokenized dataset. Sample columns: {tokenized_dataset.column_names}")
    
    total_loss = 0
    total_tokens = 0
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, 
        return_tensors="pt"
    )
    
    # Create a simple dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=4, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    print("📊 Evaluating model...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 100 == 0:
                print(f"  Processing batch {i}/{len(dataloader)}")
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Count valid tokens (not -100)
            valid_tokens = (batch['labels'] != -100).sum().item()
            total_tokens += valid_tokens
    
    # Compute average loss and perplexity
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    
    results = {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": len(dataloader)
    }
    
    return results

def evaluate_model_with_trainer(model, tokenizer, val_dataset, output_dir="./faq-model"):
    """
    Evaluate model using Trainer with proper metrics for language modeling.
    """
    from transformers import TrainingArguments
    
    # Fix: Properly tokenize the dataset
    print("📝 Tokenizing validation dataset for trainer...")
    
    def tokenize_function(examples):
        # Handle both single strings and lists of strings
        texts = examples["text"]
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # Don't pad here, let the data collator handle it
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
        
        return tokenized
    
    # Tokenize the dataset
    tokenized_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing for trainer"
    )
    
    # Setup minimal training args for evaluation
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=4,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        report_to=None,
        logging_dir=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, 
        return_tensors="pt"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_perplexity
    )
    
    # Evaluate
    results = trainer.evaluate()
    
    return results

def generate_sample_responses(model, tokenizer, test_questions, max_length=256):
    """
    Generate sample responses to test questions.
    """
    model.eval()
    device = next(model.parameters()).device
    
    print("\n🤖 Sample Model Responses:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions[:3]):  # Test first 3 questions
        # Format the question
        prompt = f"### Question: {question}\n### Answer:"
        
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "### Answer:" in response:
            answer = response.split("### Answer:")[-1].strip()
        else:
            answer = response
        
        print(f"\nQuestion {i+1}: {question}")
        print(f"Answer: {answer}")
        print("-" * 50)

def inspect_dataset(dataset, name="Dataset"):
    """
    Helper function to inspect the dataset structure
    """
    print(f"\n🔍 Inspecting {name}:")
    print(f"  Size: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Features: {dataset.features}")
    
    # Show a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"    {key}: '{value[:100]}...' (length: {len(value)})")
            else:
                print(f"    {key}: {type(value)} - {value}")

def main():
    """Main evaluation script"""
    print("=" * 50)
    print("FAQ Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_dir = "./faq-model"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found. Please train the model first.")
        return
    
    # Check if test data exists
    if not os.path.exists("./data/test"):
        print("Error: Test data not found. Please run data_preprocessing.py first.")
        return
    
    try:
        # Load model and tokenizer
        print("📥 Loading model and tokenizer...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        # Load test dataset
        print("📥 Loading test dataset...")
        test_dataset = load_from_disk("./data/test")
        
        # Inspect the dataset
        inspect_dataset(test_dataset, "Test Dataset")
        
        # Method 1: Simple evaluation
        print("\n🔍 Method 1: Simple Evaluation")
        simple_results = evaluate_model_simple(model, tokenizer, test_dataset)
        
        print("\n📊 Simple Evaluation Results:")
        for key, value in simple_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Method 2: Trainer evaluation (optional)
        print("\n🔍 Method 2: Trainer Evaluation")
        try:
            trainer_results = evaluate_model_with_trainer(model, tokenizer, test_dataset)
            print("\n📊 Trainer Evaluation Results:")
            for key, value in trainer_results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        except Exception as e:
            print(f"Trainer evaluation failed: {e}")
            trainer_results = None
        
        # Method 3: Generate sample responses
        print("\n🔍 Method 3: Sample Response Generation")
        
        # Get some test questions
        test_questions = [
            "How do I reset my password?",
            "What payment methods do you accept?",
            "How do I cancel my subscription?"
        ]
        
        generate_sample_responses(model, tokenizer, test_questions)
        
        # Save evaluation results
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": model_dir,
            "test_dataset_size": len(test_dataset),
            "simple_evaluation": simple_results,
            "trainer_evaluation": trainer_results,
        }
        
        with open(f"{model_dir}/evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n✅ Evaluation completed! Results saved to {model_dir}/evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()