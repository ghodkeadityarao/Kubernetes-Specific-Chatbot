# Fine-Tuning a Language Model for Kubernetes Documentation Queries

## Overview
This project fine-tunes a pre-trained language model to improve responses for Kubernetes documentation-related queries. It involves scraping Kubernetes documentation, creating a dataset, and training a transformer-based model to enhance response quality.

## Features
- **Web Scraping:** Extracts content from Kubernetes documentation pages.
- **Dataset Preparation:** Converts scraped content into a structured dataset suitable for fine-tuning.
- **Model Fine-Tuning:** Utilizes `transformers`, `bitsandbytes`, and `trl` libraries to fine-tune a `microsoft/phi-2` model for question-answering tasks.
- **Efficient Training:** Uses quantization techniques to optimize performance and memory usage.

## Installation
Before running the code, install the necessary dependencies using:
```bash
!pip install -q -U bitsandbytes transformers peft accelerate datasets scipy einops evaluate trl rouge_score
```

The code might take longer time to run and requires GPU. You can use Google Colab for the .ipynb file.

## Workflow
### 1. Authentication
To access Hugging Face models, login is required:
```python
from huggingface_hub import interpreter_login
interpreter_login()
```

### 2. Web Scraping Kubernetes Documentation
The script retrieves all Kubernetes documentation pages and extracts relevant text:
```python
def get_all_kubernetes_pages():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        pages = [urljoin(BASE_URL, link['href']) for link in links if link['href'].startswith('/docs/')]
        return list(set(pages))
    return []
```

### 3. Dataset Creation
The scraped text is formatted into a dataset containing "question" and "answer" pairs:
```python
def scrape_kubernetes_docs(force_refresh=False):
    if os.path.exists(CACHE_FILE) and not force_refresh:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    dataset = []
    for url in get_all_kubernetes_pages():
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            dataset.append({"question": f"What is discussed on {url}?", "answer": page_text})
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    return dataset
```

### 4. Model Loading and Quantization
To optimize memory usage, a 4-bit quantized model is loaded:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model_name = 'microsoft/phi-2'
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)
```

### 5. Tokenization
The tokenizer is initialized and configured:
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side="left",
    add_eos_token=True, add_bos_token=True, use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token
```

## Expected Output
Once the dataset is created, it will be split into `train`, `test`, and `eval` sets:
```python
DatasetDict({
    train: Dataset({ features: ['question', 'answer'], num_rows: 616 }),
    test: Dataset({ features: ['question', 'answer'], num_rows: 78 }),
    eval: Dataset({ features: ['question', 'answer'], num_rows: 78 })
})
```

## Usage
- **Scrape documentation**: `scrape_kubernetes_docs()`
- **Prepare dataset**: `prepare_finetuning_dataset()`
- **Fine-tune model**: Implement training using `SFTTrainer`

## Dependencies
- Python 3.11+
- `transformers`
- `datasets`
- `bitsandbytes`
- `trl`
- `huggingface_hub`
- `beautifulsoup4`
- `requests`

## Conclusion
This project enables the fine-tuning of a transformer-based model for Kubernetes-related queries using an automatically generated dataset. It leverages web scraping, dataset creation, and model quantization for an optimized training pipeline.

