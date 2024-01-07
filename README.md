# English-Japanese & Japanese-English NLP Translation

The repository contains the data pre-processing, model training and evaluation components for the final year Individual project focused on **English-Japanese** and **Japanese-English** NLP translation.

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