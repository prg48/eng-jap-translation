# eng-jap-translation

The project is model training and evaluation part of the final year **Individual Project** on **English-Japanese NLP translation**. 

## Data pre-processing
The data pre-processing scripts can be found in [data-preprocessing-scripts](/data-preprocessing-scripts/). The data pre-processing was first tried on local PC using the [JparaCrawlv2.0](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/), [JparaCrawlv3.0](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/), and [JESC](https://nlp.stanford.edu/projects/jesc/) corpuses. However, due to the large sizes of the files, it could not be pre-processed on the local PC. Thus, **Goole Colab Pro** was used as it provided **80GB RAM** and **40GB GPU** capability. 

The main libraries used for data pre-processing is the **huggingface API** which provided easy to use datasets library. Apart from huggingface API, **pandas** and **xmltodict** libraries were also used. The pre-processed data was stored in **Google Drive** as it provided a convenient storage to load and save data in google colab.

The data pre-processing was conducted in two stages. Firstly, corpuses were transitioned to a **huggingface datasets** format with its original columns. The implmentation can be found in [Dataset Creation JparaCrawl.ipynb](/data-preprocessing-scripts/Dataset%20Creation%20JParaCrawl.ipynb), and [Dataset Creation JESC.ipynb](/data-preprocessing-scripts/Dataset%20Creation%20JESC.ipynb). Then for the second stage, the saved datasets were loaded, unwanted columns removed, and different versions of the datasets containing different amount of sentence pairs with **train**, **validation**, and **test** splits were saved on the basis of **score filtering** or **random subset**. The implementation for this stage can be found in [Dataset Cleaning JparaCrawl.ipynb](/data-preprocessing-scripts/Dataset%20Cleaning%20JParaCrawl.ipynb). and [Dataset Cleaning JESC.ipynb](/data-preprocessing-scripts/Dataset%20Cleaning%20JESC.ipynb).

Different versions of datasets were created as fine-tuning on very large datasets requires huge computational resources which can be very costly on monetary terms with several iterations of fune-tuning adding to the costs. Therefore, subset of **JParaCrawl** data were selected based on **score** of sentence quality provided in the original data itself. And for the **JESC** data, random subset was chosen. The following table shows the data that was considered for training and their respective **train**, **validation**, and **test** splits.

|    | Train | Validation | Test |
|----|-------|------------|------|
|JParacrawlv2.0 score over 75 | 3,048,105 | 376,310 | 338,679 |
|JparaCrawlv2.0 score over 78 | 518,832 | 64,854 | 64,855 |
|JParaCrawlv3.0 score over 77 | 1,503,020 | 187,877 | 187,878 |
| JESC original | 2,237,910 | 279,739 | 279,739 |
| JESC random select | 800,000 | 100,000 | 100,000 |