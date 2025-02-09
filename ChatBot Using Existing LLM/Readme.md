# Chat with Kubernetes Docs

This project enables users to interact with Kubernetes documentation in a conversational way. It uses Google's Gemini model for Q&A and a FAISS-based vector store for efficient similarity search over Kubernetes documentation.

## Features
- Scrapes Kubernetes documentation from official Kubernetes website.
- Breaks down the scraped content into chunks for embedding.
- Uses the FAISS vector store to index the content.
- Uses a custom Q&A model based on Google Generative AI (Gemini) to answer user queries.
- Streamlit frontend for interacting with the chatbot interface.

## Prerequisites
Before running the application, make sure you have the following:
- Python 3.x
- Google Cloud API Key with access to Google's Generative AI models
- Necessary Python libraries

## Setup and Installation

1. **Clone the Repository:**
