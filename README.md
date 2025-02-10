# Kubernetes Specific Chatbot

This repository contains resources for fine-tuning a lightweight LLM (1-3B parameters) to improve responses for Kubernetes documentation-related queries. The project is structured into multiple folders, each serving a specific purpose.

## Repository Structure

### 1. **Chatbot with Existing LLM (Gemini)**
   - This folder contains a chatbot implementation using the pre-trained Gemini model.
   - Includes:
     - **Pre-built Model**: Gemini-based chatbot for Kubernetes queries.
     - **Inference Scripts**: Scripts to interact with the existing model.
     - **Example Queries**: Sample inputs and expected responses.

### 2. **Fine-Tuning**
   - This folder contains resources for fine-tuning the LLM on Kubernetes-specific data.
   - Includes:
     - **Training Scripts**: Scripts to fine-tune the model using domain-specific data.
     - **Hyperparameter Configs**: JSON files defining model training parameters.
     - **Checkpoints**: Saved model weights for evaluation and inference.

Detailed explaination is given in each folder in the respective Readme.md file.
