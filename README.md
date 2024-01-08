# English-Japanese & Japanese-English NLP Translation

The repository contains the data pre-processing, model training and evaluation components for the final year Individual project focused on **English-Japanese** and **Japanese-English** NLP translation. The final English-Japanese translation model can be checked in [english-japanese model link for huggingface](https://huggingface.co/Prgrg/en-ja-v4.0) and [japanese-english model link for huggingface](https://huggingface.co/Prgrg/ja-en-dataset-v3.0-subset-v3.0) for Japanese-English traslation.

## Table of Contents

- [Data pre-processing](#data-pre-processing)
  - [Libraries and Storage](#libraries-and-storage)
  - [Pre-processing Steps](#pre-processing-steps)
  - [Dataset Versions and Splits](#dataset-versions-and-splits)
- [Model Fine Tuning](#model-fine-tuning)
  - [Selection of Pre-trained Models](#selection-of-pre-trained-models)
  - [Computational Resources and Initial Experiments](#computational-resources-and-initial-experiments)
  - [Fine-Tuning Procedure](#fine-tuning-procedure)
  - [Training Scripts and Deployment](#training-scripts-and-deployment)

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

## Evaluation

Both models for English-Japanese and Japanese-English translations are **Marian MT** models. The models were iteratively trained for small epochs on the datasets. The English-Japanese base Marian model was trained in 4 separate sessions. They are as shown in the following table:

| model | base model | initial lr | optimizer | lr decay | epochs | dataset |
|-------|------------|------------|----------|--------|------|--------|
| ja-en-v1.0| Marian base | 0.0005 | Adam W | Poly-Decay | 2 | JParaCrawlv2.0 (score > 78) |
| ja-en-v2.0| ja-en-v1.0 | 0.0005 | Adam W | Poly-Decay | 2 | JParaCrawlv2.0 (score > 78) |
| ja-en-v3.0| ja-en-v2.0 | 0.0005 | Adam W | Poly-Decay | 4 | JParaCrawlv2.0 (score > 78) |
| ja-en-v4.0| ja-en-v3.0 | 0.0005 | Adam W | Poly-Decay | 10 | JParaCrawlv2.0 (score > 78) |

The graph of training vs validation loss for all the training sessions combined is as follows:

| ![eng-jap train vs validation loss](/images/eng-jap-training-and-validation-loss.png) |
| :------------------------------------------------: |
| Eng-Jap train vs validation loss                              |

Throughout the training process, the training loss decreases steadily, indicating that the model is improving its understanding of the dataset and learning to map English sentences to their corresponding Japanese translations. On the other hand, the validation loss follows a more complex pattern. Initially, the validation loss decreases, reaching its lowest point at epoch 4. This suggests that the model is generalizing well to unseen data. However, between epochs 4 and 5, the validation loss increases dramatically, which could be a sign of overfitting. After epoch 9, the validation loss starts to decrease again, and the trend continues until the end of the training process, although with some fluctuations. This indicates that the model is once again learning to generalize better the validation dataset, although not as effectively as during the initial epochs. However, even though the model was generalizing well initially, the base model had a very low BLEU score on the validation dataset which will be discussed in later section. However, after 18 epochs, the quality of translation as well as the BLEU score and BERT score improved significantly. Thus, the final version of the model after 18 epochs of fine-tuning was chosen as the final model for the English-Japanese Translation.

However, unline the English-Japanese model, the Japanese-English model required a lot of experimentation to get the desired results. The model was trained on 4 our of 5 datasets mentioned in [Datasets versions and splits](#dataset-versions-and-splits). As can be seen in the following table, it was trained for a total of 11 sessions with some sessions running on different hyper-parameters, base models and datasets.

| model | base model | initial lr | optimizer | lr decay | epochs | dataset |
|------|------|-------|------|------|-------|------|
| ja-en-v1.0 | Marian base | 0.0005 | Adam W | PolyDecay | 2 | JparaCrawlv2.0 (score > 78) |
| ja-en-JESC-v1.0 | Marian base | 0.0005 | Adam W | PolyDecay | 4 | JESC (random select) |
| ja-en-JESC-v2.0 | ja-en-JESC-v1.0 | 0.0005 | Adam W | PolyDecay | 4 | JESC (random select) |
| ja-en-JESC-full-v1.0 | Marian base | 0.0005 | Adam W | PolyDecay | 6 | JESC (original) |
| ja-en-JESC-v3.0 | ja-en-JESC-v2.0  | 0.0005 | Adam W | PolyDecay | 6 | JESC (random select) |
| ja-en-JESC-v3.0-wth-lr-decay | ja-en-JESC-v2.0  | 0.0001 | Adam W | PolyDecay | 10 | JESC (random select) |
| ja-en-JESC-v4.0-wth-lr-decay | ja-en-JESC-v3.0-wth-lr-decay  | 0.0002 | Adam W | PolyDecay | 8 | JESC (random select) |
| ja-en-JESC-v5.0-wth-lr-decay | ja-en-JESC-v4.0-wth-lr-decay  | 0.0001 | Adam W | PolyDecay | 6 | JESC (random select) |
| ja-en-dataset-v3.0-subset-v1.0 | Marian base  | 0.0005 | Adam | LinWarmup-CosSchedule | 2 | JParaCrawlv3.0 (score > 77) |
| ja-en-dataset-v3.0-subset-v2.0 | ja-en-dataset-v3.0-subset-v1.0 | 0.0005 | Adam | LinWarmup-CosSchedule | 7 | JParaCrawlv3.0 (score > 77) |
| ja-en-dataset-v3.0-subset-v3.0 | ja-en-dataset-v3.0-subset-v2.0 | 0.0001 | Adam | LinWarmup-CosSchedule | 1 | JParaCrawlv3.0 (score > 77) |

The graphs of training vs validation loss for japanese-english sessions which can be grouped in 5 separate sessions is shown in the figure below:

| ![jap-eng train vs validation loss](/images/jap-eng-training-and-validation-loss.png) |
| :------------------------------------------------: |
| Jap-Eng train vs validation losses for different model sessions                              |

1. **JParaCrawl v2.0 score over 78**: The training loss decreases significantly, but the validation loss is high and increases over time. This is a sign of overfitting, as the model is performing well on the training data but not generalizing well 