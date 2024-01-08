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

Both models for English-Japanese and Japanese-English translations are **Marian MT** models. The models were iteratively trained for small epochs on the datasets on different sessions.

The English-Japanese model was trained on different sessions using the Marian MT model as the base model and each iterative session using the previous session model as base session. The final model metrics is shown in the following table:

| model | base model | initial lr | optimizer | lr decay | epochs | dataset |
|-------|------------|------------|----------|--------|------|--------|
| en-ja-v4.0| Marian base | 0.0005 | Adam W | Poly-Decay | 18 | JParaCrawlv2.0 (score > 78) |

The graph of training vs validation loss for all the model is as follows:

| ![eng-jap train vs validation loss](/images/eng-jap-training-and-validation-loss.png) |
| :------------------------------------------------: |
| Eng-Jap train vs validation loss                              |

Throughout the training process, the training loss decreases steadily, indicating that the model is improving its understanding of the dataset and learning to map English sentences to their corresponding Japanese translations. On the other hand, the validation loss follows a more complex pattern. Initially, the validation loss decreases, reaching its lowest point at epoch 4. This suggests that the model is generalizing well to unseen data. However, between epochs 4 and 5, the validation loss increases dramatically, which could be a sign of overfitting. After epoch 9, the validation loss starts to decrease again, and the trend continues until the end of the training process, although with some fluctuations. This indicates that the model is once again learning to generalize better the validation dataset, although not as effectively as during the initial epochs. However, even though the model was generalizing well initially, the base model had a very low BLEU score on the validation dataset which will be discussed in later section. However, after 18 epochs, the quality of translation as well as the BLEU score and BERT score improved significantly. Thus, the final version of the model after 18 epochs of fine-tuning was chosen as the final model for the English-Japanese Translation.

However, unline the English-Japanese model, the Japanese-English model required a lot of experimentation to get the desired results. The model was trained on 4 our of 5 datasets mentioned in [Datasets versions and splits](#dataset-versions-and-splits). As can be seen in the following table, it was trained for a total of 5 sessions with some sessions running on different hyper-parameters, base models and datasets.

| model | base model | initial lr | optimizer | lr decay | epochs | dataset |
|------|------|-------|------|------|-------|------|
| ja-en-v1.0 | Marian base | 0.0005 |Adam W | Poly Decay |2 |JparaCrawlv2.0 (score > 78)|
| ja-en-JESC-v3.0 | Marian base | 0.0005 | Adam W |Poly Decay | 14 | JESC (random select) |
| ja-en-JESC-full-v1.0 | Marian base | 0.0005 | Adam W | PolyDecay | 6 | JESC (original) |
| ja-en-JESC-v5.0-with-lr-decay | ja-en-JESC-v2.0 | 0.0001 | Adam W | Poly Decay | 19 | JESC (random select) |
| ja-en-dataset-v3.0-subset-v3.0 | Marian base | 0.0001 - 0.0005 | Adam | LinWarmup-CosSchedule | 11 | JparaCrawlv3.0 (score > 77) |

The graphs of training vs validation loss for japanese-english sessions is shown in the figure below:

| ![jap-eng train vs validation loss](/images/jap-eng-training-and-validation-loss.png) |
| :------------------------------------------------: |
| Jap-Eng train vs validation losses for different model sessions                              |

1. **ja-en-v1.0**: This model shows a sharp increase in validation loss after the first epoch, suggesting that the model may not be generalizing well beyond the training data. The train loss is relatively stable, but the significant rise in validation loss indicates potential overfitting.

2. **ja-en-JESC-v3.0**: The training loss for this model generally decreases, indicating learning and improvement. However, there's a marked increase in validation loss after the 8th epoch, which points to overfitting as the model learns the training data too closely at the expense of its ability to generalize.

3. **ja-en-JESC-full-v1.0**: The train and validation losses here decreases together initially, which is a good indicator of the model learning effectively. The losses start to plateau toward the end of the available epochs, suggesting the model might be beginning to converge.

4. **ja-en-JESC-v5.0-with-lr-decay**: This model's train and validation losses decrease over time, displaying a consistent learning pattern without significant overfitting. The learning rate decay seems to be effective in controlling the learning process, yet there are signs of slight overfitting after the 4th spoch as the validation loss starts to increase slightly while the training loss continues to decrease.

5. **ja-en-dataset-v3.0-subset-v3.0**: The training loss shows a consistent decline, which is an ideal scenario as it suggests the model is effectively learning the translation task. The validation loss also decreases and remains low, which indicates good generalization to unseen data. This model demonstrates a balance between learning and generalizing without the apparent overfitting seen in other models.

The **ja-en-v3.0-subset-v3.0** model was chosen as the final model for several reasons:

* **Steady Learning Curve**: The model showed a steady decrease in training loss, which is an indicator of stable learning without drastic fluctuations that could suggest instability in the learning process.

* **Low and Stable Validation Loss**: The validation loss not only decreased alongside the training loss but also stabilized at a low level. This pattern is indicative of good generalization capabilities, suggesting that the model is not just memorizing the training data but is learning to translate new, unseen sentences effectively.

* **No overfitting**: Unlike other models, this model did not exhibit a significant invrease in validation loss at any point during training, which would be a sign of overfitting.

* **Efficient Learning**: Considering the computational resources and time constraints, this model provided a favorable balance between performance and efficiency, reaching satisfactory loss levels within a reasonable number of epochs

The chosen English-Japanese and Japanese-English models were both evaluated on **BLEU** and **BERT** scores for the validation as well as test data.

The metrics for validation data is as follows:

| Model | BLEU | bert-score |
|-------|------|------------|
|Eng-Jap Marian base | 0.66 | 0.60 |
|en-ja-v4.0 | 37.41 | 0.84 |
| Jap-Eng Marian base | 0.14 | 0.78 |
| ja-en-v3.0-subset-v3.0 | 19.68 | 0.92 |

The metrics for test data is as follows:

| Model | BLEU | bert-score |
|-------|------|------------|
|Eng-Jap Marian base | 0.68 | 0.61 |
|en-ja-v4.0 | 37.71 | 0.84 |
| Jap-Eng Marian base | 0.15 | 0.77 |
| ja-en-v3.0-subset-v3.0 | 19.71 | 0.92 |

As seen from above tables the fine-tuned models demonstrate significant improvements over their respective base models in both BLEU and BERT score metrics. These improvements validate our choice of fine-tuning with the chosen datasets and indicate that our models have better translation quality than the base models. However, While the English-Japanese fine-tuned model shows a considerable increase in BLEU score, the Japanese-English fine-tuned model's increase is relatively lower. this discrepancy suggests that there might be room for furhter optimization or exploration of other architectures to improve the Japanese-English translation quality. Also, although the bert-score has improved for both fine-tuned models, it is important to note that these metrics do not fully capture the translation quality's nuances, such as fluency which might require human evaluation for more comprehensive assenssment of the translation performance.

Here are some examples of the base models and the fine-tuned model's performance on actual sentences on different domains:

English to Japanese (accuracy of translated sentence checked according to google translate and wrapped in parenthesis after the respective translated japanese sentence)

| sentence | Marian base model translation | en-ja-v4.0 translation |
|----------|---------------------|-------|
|The latest smartphone model features an advanced neural processor for optimized artificial intelligence tasks. | (これ から は , かげ の ため を 耕 す 人 は , クミン を 耕 す 者 と 思 い , はやかげ の ため に , 定め られ て い る .) (From now on, a person who cultivates a field for the sake of darkness will be considered a person who cultivates cumin, and it is prescribed for the sake of the forest.) | 最新のスマートフンモデルは人工知能タスクを最適化するための高度なニューラルプロセッサーを備えています (The latest smart hun model features an advanced neural processor to optimize artificial intelligence tasks)|
|Regular exercise contributes significantly to cardiovascular health and overall well-being.| 忍耐 は 錬達 を 回復 し , クモナ は 良 い 一生 を 経 て , 記憶 を 経 た 者 で あ る .(Patience restores strength, and Kumona is a person who has lived a good life and gained memories.) | 定期的な運動は心血管の健康と全体的な福利に大きく貢献します. (Regular exercise greatly contributes to cardiovascular health and overall well-being.) |
|Investors are diversifying their portfolios to mitigate risks amid fluctuating market conditions.| し わ れ る 場合 に は , 彼 ら の 交わり が あ っ て , クレトモス に 寄港 する こと に な っ て い る . (In case of trouble, their association is such that they should call at Clethmos.) | 投資家はポートフリオを多様化させリスクの変動市場の状を軽減しています (Investors are diversifying their portfolios to reduce the risk of volatile market conditions) |
|Cognitive learning theories emphasize the role of mental processes in understanding and retaining new information.| すぐれた学 者 は さと い わ な に すぎ な い こと を さ し て も , さとき 者 は さと る こと が たやす く , 新し い こと に よ っ て 解 か れ る . (Even if a good scholar points out something that is only a trivial thing, a discerning person can find it easily and solve it based on something new.) | 認知学習理論は新しい情報の理解と保持におけるメンタルプロセスの役割を強調しています (Cognitive learning theory emphasizes the role of mental processes in understanding and retaining new information) |
|The ancient ruins of Machu Picchu in Peru offer breathtaking views and a glimpse into Incan history.|古 い 境 は キシび と い う 物 の 荒れ 跡 が これ を ささげ る こと に よ っ て , その 境 に 達 し た もの で あ る . (The old boundary was reached by the desolate ruins of something called Kisubi, which were dedicated to it.) | ペルーのマチュピチュピチュの古代遺跡は息をめた景色とインデントに垣間見渡せます (The ancient ruins of Peru's Machu Picchu Picchu offer breathtaking views and indented glimpses) |

Japanese to English (The Japanese texts were taken from google translate and the english translation that was entered in google translate to get the japanese texts is enclosed in paranthesis).

| sentence | Marian base model translation | ja-en-dataset-v3.0-subset-v3.0 |
|----------|--------------------------------|-------------------|
|最新のスマートフォン モデルには、人工知能タスクを最適化するための高度なニューラル プロセッサが搭載されています。(The latest smartphone model features an advanced neural processor for optimized artificial intelligence tasks.) | For the beginning of good S land of yield yield yield yield yield yield yield yield yield yield yield yield as far to land land as far yield yield yield yield yield sail sail sail sail sailee set set set sail time time time time time time as far excellent S so need sail according to men; | The latest smartphone model is equipped with advanced neural processors to optimize artifi intelligence tasks.|
|定期的な運動は、心臓血管の健康と全体的な幸福に大きく貢献します。 (Regular exercise contributes significantly to cardiovascular health and overall well-being.) | but the manifestationmentmentment there is good indeed to be possessed possessed by the body to the far farment of a full gift for every good pleasure in the render himself himself in the fulness of blood. | The regular exercise contributes significantly to the health and overall wellbeing of the cardiovascular. |
|投資家は、市場環境が変動する中、リスクを軽減するためにポートフォリオを多様化しています。 (Investors are diversifying their portfolios to mitigate risks amid fluctuating market conditions.) | And for the house we have sailed dwelling dwellingly in the an house house in the called listen listen listen to the holy house Syy by dead dead dead dead dead dead dead dead Syyyyyyyy by land land land land land land land land land land land land land according to the holy far far far ...{truncated} | Investors have diversified their portfolio to alleviate their risks as market conditions fluctuate. |
|認知学習理論では、新しい情報を理解して保持する際の精神的プロセスの役割を強調しています。 (Cognitive learning theories emphasize the role of mental processes in understanding and retaining new information.) | And that ye have ought ought ought ought ought ought ought ought ought ought ought ought ought ought oughtyyy to knowledge, carried to knowledge knowledge knowledge knowledge knowledgeyyyyyyyyy knowledge knowledge; | "The cognitive learning theory emsizes the role of a psylogical process in understanding and retaining new information. |
| ペルーのマチュ ピチュの古代遺跡では、息を呑むような景色とインカの歴史を垣間見ることができます。(The ancient ruins of Machu Picchu in Peru offer breathtaking views and a glimpse into Incan history.) | And at the age of the ancients of offerings, as the shadow of living living living living living living living height, as the four sons of a remnant remnant remnant to a young eagle, as the sons of alt among the sons of alt among the sons of silver among the sons of alt among the sons of alt among herself. | In the ancient ruins of Machu Pigu in Peru, you can glimpses of breathtaking scenery and the history of Inca. |