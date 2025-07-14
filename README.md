# FAQ-Chatbot-using-Fine-tuned-Language-Models

This project implements an end-to-end FAQ chatbot system using a fine-tuned GPT-2 model on customer support Q&A pairs. It leverages the Hugging Face Transformers library and PyTorch to train and deploy a conversational AI assistant for FAQs.

## 📌 Features

- Fine-tuned GPT-2 on 17.5K curated FAQ pairs.
- Custom training loop with learning rate scheduling, gradient clipping, and evaluation.
- Supports training/evaluation on large datasets (87.6K samples).
- Generates conversational responses to user queries.

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Datasets (🤗)
- Weights & Biases (optional for tracking)

🔧 Setup Instructions
After cloning the repository, follow the steps below to get started:

1. Install Dependencies
pip install -r requirements.txt
Note: Ensure you are using Python 3.9, as some libraries used in this project may be incompatible with newer versions.

2. Run the Code
Once dependencies are installed, you can execute the training, evaluation, and inference scripts directly.

⚠️ Current Status:
The model has been trained on 2.5% of the SQuAD dataset for demonstration purposes. An updated version of the code with extended dataset usage and improved performance will be committed soon.
