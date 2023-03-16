# A simple baseline for the Kaggle LECR

🔗  [Kaggle-Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/overview)





👋 Hi there!

🔍 Here's a simple baseline I ran for this competition, because I joined the game quite late.⏰ I only had time to run a quick code, but it was still a lot of fun! 😄

This is a Natural Language Processing (NLP) competition that focuses on semantic matching for text. However, it also involves many concepts and techniques from recommendation systems, which are worth exploring. The competition has two stages: **Retriever** and **Reranker**.



### Stage 1: Retriever 📖

In the retriever stage, the goal is to retrieve as many `content_id` as possible for each `topic_id`. Then create a new dataset `top_n.csv`, and determining whether each `content_id` in the retrieved set is part of the `ground truth` or not. A label of 0 or 1 is assigned accordingly.

### Stage 2: Reranker 🔍

In the re-ranker stage, the aim is to determine which of the retrieved content IDs should be kept.



## 💻 Usage

To run the code for both stages, use the following command:

```
python stage1/Main.py
python stage2/Main.py
```



The code structure is as follows:

```

|-- data
|   |-- raw
|   |   |-- content.csv
|   |   |-- correlations.csv
|   |   |-- sample_submission.csv
|   |   |-- topics.csv
|-- logs
|   |-- ex.log
|-- outputs
|   |-- ex
|   |   |-- oof_df.csv
|   |   |-- xx_fold0.pth
|   |   |-- train.log
|   |   |-- train_top20.csv
|   |   |-- train_top5.csv
|   |   |-- train_top50.csv
|-- src
|   |-- stage1
|   |   |-- Config.py
|   |   |-- Dataset.py
|   |   |-- Main.py
|   |   |-- Metric.py
|   |   |-- Model.py
|   |   |-- Train.py
|   |   |-- Utils.py
|   |-- stage2
|   |   |-- Config.py
|   |   |-- Dataset.py
|   |   |-- Main.py
|   |   |-- Model.py
|   |   |-- Train.py
|   |   |-- Utils.py
|   |   |-- generate_data.ipynb

```



