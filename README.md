# English-Japanese & Japanese-English NLP Translation

The repository contains the data pre-processing, model training and evaluation components for the final year Individual project focused on **English-Japanese** and **Japanese-English** NLP translation. The final English-Japanese translation model can be checked in [english-japanese model link for huggingface](https://huggingface.co/Prgrg/en-ja-v4.0) and [japanese-english model link for huggingface](https://huggingface.co/Prgrg/ja-en-dataset-v3.0-subset-v3.0) for Japanese-English traslation.

## Data pre-processing

Efficient data pre-processing is crucial for the success of NLP models. The scripts employed in the data pre-processing stages for the project can be accessed via the [data-preprocessing-scripts](/data-preprocessing-scripts/) directory. Despite initial attempts to process data locally using JparaCrawl and JESC corpuses, the considerable file sizes necessitated the use of **Google Colab Pro**'s capabilities, including **80GB RAM** and **40GB GPU**.

### Libraries and Storage

The project leveraged the **Hugging Face API** for its robust datasets library, facilitating easier manipulation and preparation for the data. Alongside Higging Face, **Pandas** for data manipulation and **xmltodict** for XML parsing were integral to our pre-processing. The processed data was stored in **Google Drive**, offering a seamless integration with **Google Colab** for efficient loading and storage operations.

### Pre-processing Steps

The data pre-processing unfolded in two primary stages:

1. **Initial Conversion to Hugging Face Datasets**: The corpuses were initially converted to a Hugging Face datasets format, maintaining the original column structure. The scripts for this stage are available in [Dataset Creation JparaCrawl.ipynb](/data-preprocessing-scripts/Dataset%20Cleaning%20JParaCrawl.ipynb) and [Dataset Creation JESC.ipynb](/data-preprocessing-scripts/Dataset%20Creation%20JESC.ipynb).

2. **Data Cleaning and Subset Creation**: Post conversion, datasets were further refined by removing unwanted columns and creating various subsets of the data with different sentence pair quantities. These subsets were classified into **train**, **validation**, and **test** splits, determined by either **score filtering** or selecting a **random subset**. The detailed implementation can be found in [Dataset Cleaning JparaCrawl.ipynb](/data-preprocessing-scripts/Dataset%20Cleaning%20JParaCrawl.ipynb) and [Dataset Cleaning JESC.ipynb](/data-preprocessing-scripts/Dataset%20Cleaning%20JESC.ipynb).

### Dataset Versions and Splits

Given the computational demands and associated costs of fine-tuning on extensive datasets, subsets of **JparaCrawl** data were curated based on the sentence quality scores from the original data. Meanwhile, a random subset was selected from the **JESC** data. The table below details the training splits and their respective sizes:

|  Dataset  | Train | Validation | Test |
|----|-------|------------|------|
|JParacrawlv2.0 (score > 75) | 3,048,105 | 376,310 | 338,679 |
|JparaCrawlv2.0 (score > 78) | 518,832 | 64,854 | 64,855 |
|JParaCrawlv3.0 (score > 77) | 1,503,020 | 187,877 | 187,878 |
| JESC (original) | 2,237,910 | 279,739 | 279,739 |
| JESC (random select) | 800,000 | 100,000 | 100,000 |

## Model Fine Tuning

Initially, the project considered the ambitious task of coding a transformer model from scratch on the seminal paper by [Vaswani et al.](https://arxiv.org/pdf/1706.03762.pdf). however, given the intricate knowledge required of the architecture, optimization techniques, and implementation details, along with the substantial computational resources and costs associated with training a model from scratch, this approach was deemed impractical. Instead the project pivoted to fine-tuning a pre-trained model to leverage state-of-the art performance while ensuring efficiency and cost-effectiveness.

### Selection of Pre-trained Models

Among the myriad of pre-trained models available from Tensorflow Hub, OpenAI, and Hugging Face the project selected the **Marian MT** model from Hugging Face for fine-tuning. This decision was based on the user-friendly library, robust community support, and the model's specific design and optimization for machine translation tasks. The selected models were [Helsinki-NLP/opus-mt-en-jap](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap) for English-Japanese and [Helsinki-NLP/opus-mt-jap-en](https://huggingface.co/Helsinki-NLP/opus-mt-jap-en) for Japanese-English translations, both pre-trained on the **Opus dataset**.

### Computational Resources and Initial Experiments

Initial fine-tuning experiments were conducted on **Google Colab** with a small dataset to ensure feasibility before scaling up to larger datasets. Subsequent intensive training was performed on a virtual machine on **Lambda Cloud**, equipped with an **A10 24GB GPU**. Previous attempts on other cloud platforms like **Google Cloud** and **AWS** encountered issues with manual configurations and compatibility, which were efficiently resolved using Lambda Cloud's pre-installed **Lambda Stack** (Tensorflow, PyTorch and CUDA configured), facilitating a smoother execution of scripts.

### Fine-Tuning Procedure

The fine-tuning involved several critical steps, utilizing both **PyTorch** and **TensorFlow** libraries. The procedure began by loading the relevant dataset, tokenizer, model, and data collator. After tokenizing the dataset and converting it into a TensorFlow format, the model was compiled with appropriate **optimizer** and **loss function** settings, followed by fine-tuning for a specified number of epochs. Throughout the process, hyperparameters such as **max length**, **batch size**, **learning rate**, and **weight decay rate** were adjusted based on performance. Callbacks like **TensorBoard** and **PushToHubCallback** were employed for monitoring and versioning.

### Training Scripts and Deployment

The project includes two main scripts for model training, [train_en_jp.py](/training-scripts/train_en_jp.py) for English-Japanese and [train_jp_en.py](/training-scripts/train_jp_en.py) for Japanese-English translations. Upon completion of the fine-tuning, the most efficient versions of the models, as determined by **BLEU** and **BERT** scores, were chosen for deployment. The final models for both English-Japanese and Japanese-English translations were deployed in huggingface and can be checked through [english-japanese model link](https://huggingface.co/Prgrg/en-ja-v4.0) and [japanese-english model link](https://huggingface.co/Prgrg/ja-en-dataset-v3.0-subset-v3.0).

## Model Evaluation

Evaluating the performance of our fine-tuned models is crucial to ensuring the accuracy and reliability of translations provided by the system.

### Selections and Metrics

Out of 15 fine-tuned versions for each language pair, models were selected based on their performance evaluated by **BLEU** and **BERT** scores. These metrics are pivotal in machine translation as they provide an objective measure of the translation's quality compared to the reference text.

### Evaluation Process

The evaluation was conducted systematically to ensure a thorough assessment of each model's capabilities:

1. **Loading Resources**: The process commenced with the loading of the relevant dataset, checkpoint, tokenizer, and the respective fine-tuned or base model.
2. **Tokenization**: Each dataset was tokenized to convert the text into a format suitable for the model's understanding.
3. **Data Conversion**: The dataset split destined for evaluation was converted into a TensorFlow data format, compatible with the model's architecture.
4. **Prediction and Decoding**: The models were then used to generate predictions from the data, which were subsequently decoded and compiled into a list for comparision.
5. **Score Calculation**: The decoded predictions and actual labels were passed to the compute functions of **BLEU** and **BERT** scores to evaluate the model's performance quantitatively.