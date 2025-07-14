import pandas as pd
import json
import re
import os
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
    return text.strip()

def load_faq_dataset():
    """Load FAQ dataset from Hugging Face or fallback to synthetic"""
    try:
        dataset = load_dataset("squad")
        print("✅ Loaded SQuAD dataset from Hugging Face")
        return dataset
    except Exception as e:
        print("⚠️ Failed to load SQuAD. Creating synthetic dataset instead.")
        return {"train": Dataset.from_list(create_synthetic_faq_data())}

def create_synthetic_faq_data():
    """Return a list of synthetic FAQ data"""
    return [
        {"question": "How do I create a new account?", "answer": "Visit our signup page, fill in your email and password, then verify your email address to activate your account."},
        {"question": "I forgot my password, how can I reset it?", "answer": "Click 'Forgot Password' on the login page, enter your email, and follow the reset instructions sent to your inbox."},
        {"question": "How do I delete my account permanently?", "answer": "Contact our support team at support@company.com with your account details. Account deletion takes 7-10 business days."},
        {"question": "What payment methods do you accept?", "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers."},
        {"question": "How do I cancel my subscription?", "answer": "Log into your account, go to Billing Settings, click 'Cancel Subscription'. Your access continues until the end of your billing period."},
        {"question": "Can I get a refund?", "answer": "Yes, we offer full refunds within 30 days of purchase. Contact support with your order number for processing."},
        {"question": "How do I update my billing information?", "answer": "Navigate to Account Settings > Billing, click 'Edit Payment Method', and update your card details."},
        {"question": "The application is running slowly, what should I do?", "answer": "Try clearing your browser cache, check your internet connection, and ensure you're using the latest browser version."},
        {"question": "How do I invite team members?", "answer": "In your dashboard, click 'Team Management', enter email addresses, set permissions, and send invitations."},
        {"question": "How secure is my data?", "answer": "We use enterprise-grade encryption, regular security audits, and comply with SOC 2 Type II and GDPR standards."},
        {"question": "Do you share my data with third parties?", "answer": "No, we never sell your data. We only share information as described in our Privacy Policy, primarily for service functionality."},
    ]

def prepare_training_data(dataset, test_size=0.2, val_size=0.1):
    """Prepare and split data from SQuAD or synthetic"""
    if "train" in dataset and "question" in dataset["train"].features:
        # Assume SQuAD format
        squad_data = dataset["train"]
        questions = [clean_text(q) for q in squad_data["question"]]
        answers = [clean_text(a["text"][0]) if a["text"] else "" for a in squad_data["answers"]]
    else:
        # Assume synthetic format
        questions = [clean_text(item["question"]) for item in dataset["train"]]
        answers = [clean_text(item["answer"]) for item in dataset["train"]]

    # Filter valid pairs
    valid_pairs = [(q, a) for q, a in zip(questions, answers) if q and a]
    if len(valid_pairs) < 10:
        raise ValueError("Not enough valid question-answer pairs")

    questions, answers = zip(*valid_pairs)

    # Split into train/val/test
    train_q, temp_q, train_a, temp_a = train_test_split(questions, answers, test_size=val_size + test_size, random_state=42)
    val_q, test_q, val_a, test_a = train_test_split(temp_q, temp_a, test_size=test_size / (val_size + test_size), random_state=42)

    # Return HuggingFace datasets
    return (
        Dataset.from_dict({'question': train_q, 'answer': train_a}),
        Dataset.from_dict({'question': val_q, 'answer': val_a}),
        Dataset.from_dict({'question': test_q, 'answer': test_a}),
    )

def format_for_training(example):
    """Convert question-answer to language model prompt"""
    return {
        'text': f"### Question: {example['question']}\n### Answer: {example['answer']}<|endoftext|>"
    }

def save_datasets(train_dataset, val_dataset, test_dataset, output_dir="./data"):
    """Save train/val/test datasets to disk"""
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "val"))
    test_dataset.save_to_disk(os.path.join(output_dir, "test"))

    print(f"📁 Datasets saved in {output_dir}")
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

def main():
    """Main preprocessing pipeline"""
    print("🚀 Starting data preprocessing...")
    
    dataset = load_faq_dataset()
    train_ds, val_ds, test_ds = prepare_training_data(dataset)

    train_ds = train_ds.map(format_for_training)
    val_ds = val_ds.map(format_for_training)
    test_ds = test_ds.map(format_for_training)

    save_datasets(train_ds, val_ds, test_ds)

    print("✅ Data preprocessing complete!")
    print("\n📝 Sample:")
    print(train_ds[0]["text"])

if __name__ == "__main__":
    main()
