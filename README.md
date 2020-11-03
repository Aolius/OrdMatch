# OrdMatch
Code for CIKM2019 Short Paper: [Machine Reading Comprehension: Matching and Orders](https://dl.acm.org/doi/abs/10.1145/3357384.3358139)


### Requirements
- Python 3.6
- Pytorch 1.0.1
- NLTK

### Datasets
- [RecipeQA: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes](https://arxiv.org/abs/1809.00812)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)


### Data Preprocessing
1. Download the original data files [`train.json`](https://vision.cs.hacettepe.edu.tr/files/recipeqa/train.json) and [`val.json`](https://vision.cs.hacettepe.edu.tr/files/recipeqa/val.json) to `root/data/`.
2. Download and unzip the `glove.840B.300d.zip`to `root/data/embedding/`.
3. Run ```python root/preprocess.py```. This will generate `textual_cloze_train.json` and `textual_cloze_val.json` to `root/data`. Rename them as `TC_train.json/TC_val.json`.
4. Download the [Activity Ordering data](https://drive.google.com/drive/folders/1D7r5laxXduwnBeN0DSFjqRl4BGyd3gzE?usp=sharing):
>1. AO24_train.json
>2. AO24_val.json
>3. AO8_train.json
>4. AO8_val.json          to `root/data/`.

5. download the [embedding data](https://drive.google.com/drive/folders/1d2n87K2duPAWeoLpa2yG8IBKPftjNaTw?usp=sharing)  (optional)
>1. embeddings.pt
>2. word2idx.pt
>3. idx2word.pt
>4. word2idx_count.pt    to `root/data/embedding`.

### Usage
```
python main_TC.py --batch_size 5 --lamda 0.01 #for textual cloze task
python main_AO8.py --batch_size 5 --lamda 0.01 #for AO8 task
python main_AO24.py --batch_size 5 --lamda 0.01 #for AO24 task
```
For detailed configuration, please refer to `root/config.py`.

### Hyper-parameter
For hyper-parameter `\lambda`, we recommend choosing a value between 0.01~0.1 for best results. Manul tuning of this hyperparameter is necessary.
To reproduce our results, please download the embedding data as mentioned above and follow the default configuration. 

### Citation
Please consider citing this paper if you found this project useful.

```
@inproceedings{liu2019machine,
  title={Machine Reading Comprehension: Matching and Orders},
  author={Liu, Ao and Qu, Lizhen and Lu, Junyu and Zhang, Chenbin and Xu, Zenglin},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2057--2060},
  year={2019}
}
```
### Acknowledgement
Our code is modified on top of the [comatch](https://github.com/shuohangwang/comatch) code. We thank Shuohang Wang for releasing their code.
