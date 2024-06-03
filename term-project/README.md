 # Term Project
This repository aims to reproduce the findings of the paper ["In and Out-of-Domain Text Adversarial Robustness via Label Smoothing"](https://arxiv.org/abs/2212.10258)[1]. Specifically, it focuses on analyzing the impact of label smoothing on the robustness and calibration of pre-trained language models.

The project involves:
- Implementing and fine-tuning various pre-trained language models: BERT[2], dBERT[3], RoBERTa[4].
- Applying label smoothing techniques.
- Evaluating the models' robustness and calibration against different adversarial attacks.


# Installation and Setup
To get started with this project, you will need to install the required packages listed in the `'requirements.txt'` file. If you encounter the following error:
```python
ImportError: cannot import name 'triu' from 'scipy.linalg'
```
You should downgrade `scipy` to a specific version, such as 1.10.1 for compatibility:
```python
pip install scyipy==1.10.1
```


# fine_tuning.py
`'fine_tuning.py'` is desgined to fine-tune pre-trained models on varaious text classification datasets. This script also supports two types of label smoothing, standard and adversarial label smoothing, to improve model's robustness and calibration.

## Arguments
This script accepts variout arguments to customize the fine-tuning process. Below is a detailed explanation of each argument, its purpose, and usage example.

'-m' or '--model_name' (**Required**)
- **Description**: Specifies the name of the pre-trained model to fine-tune.
- **Options**: 'bert', 'dbert', 'roberta'

'-d' or '--data_name' (**Required**)
- **Description**: Specifies the name of dataset to fine-tune the given pre-trained model.
- **Options**: 'yelp', 'ag_news'

'-b' or '--batch_size' (**Optional**)
- **Description**: Specifies the batch size for fine-tuning.
- **Default**: 64

'-n' or '--num_samples' (**Optional**)
- **Description**: Specifies the number of samples to use for fine-tuining.
- **Default**: 32768 ('2**15')

'-l' or '--label_smoothing' (**Optional**)
- **Description**: Applies label smoothing if this flag is set.
- **Default**: False (Label smoothing is not applied if this flag is not set.)

'-s' or '--smoothing_param' (**Optional**)
- **Description**: Specifies the smoothing parameter value for label smoothing.
- **Default**: 0.45

'-a' or '--adversarial' (**Optional**)
- **Description**: Applies adversarial label smoothing if this flag is set.
- **Default**: False (Standard label smooothing is applied if this flag is not set.)

## Example Command
```python
python fine_tuning.py -m bert -d yelp -l
```
This command fine-tunes the `'bert-base-uncased'` model on the `'yelp-polarity'` dataset with a batch size of 64, using 32,768 random samples from training dataset, applying standard label smoothing with a parameter of 0.45.


# clean_accuracy.py
`'clean_accurary.py'` evaluates a fine-tuned model on a specified dataset. It loads the fine-tuned model, tokenizes the randomly sampled test data, and computes the accuracy of the model. It also supports the use of label smoothing.

## Arguments
This script accepts various arguments to customize the evaluation process. Below is a detailed explanation of each argument, its purpose, and usage example.

'-m' or '--model_name' (**Required**)
- **Description**: Specifies the name of the pre-trained model to evaluate.
- **Options**: 'bert', 'dbert', 'roberta'

'-d' or '--data_name' (**Required**)
- **Description**: Specifies the name of the dataset to use for evaluation.
- **Options**: 'yelp', 'ag_news'

'-n' or '--num_samples' (**Optional**)
- **Description**: Specifies the number of samples to use for evaluation.
- **Default**: 1000

'-l' or '--label_smoothing' (**Optional**)
- **Description**: Indicates whether label smoothing was used during fine-tuning. If this flag is set, the script will load a model fine-tuned with label smoothing.
- **Default**: False (Label smoothing is not applied if this flag is not set.)

## Example Command
```python
python clean_accuracy.py -m bert -d yelp -l
```
This command evaluates the `'bert-base-uncased'` model on the `'yelp_polarity'` dataset using 1,000 randomly sampled test data and indicates that label smothing was applied during fine-tuning.

# text_attack.py
`'text_attack.py'` evaluates the robustness of a fine-tuned model using adversarial attacks on a specified dataset . It supports TextFooler[5] and BAE(BERT-based Adversarial Examples)[6].

## Arguments
This scripts accepts various arguments to customize the adversarial attack process. Below is a detailed explanation of each argument, its purpose, and usage example.

'-m' or '--model_name' (**Required**)
- **Description**: Specifies the name of the pre-trained model to evaluate.
- **Options**: 'bert', 'dbert', 'roberta'

'-d' or '--data_name' (**Required**)
- **Description**: Specifies the name of dataset to use for evaluation.
- **Options**: 'yelp', 'ag_news'

'-n' or '--num_samples' (**Optional**)
- **Description**: Specifies the number of samples to use for evaluation.
- **Default**: 1000

'-l' or '--label_smoothing' (**Optional**)
- **Description**: Indicates whether label smoothing was used during fine-tuning. If this flag is set, the script will load a model fine-tuned with label smoothing.
- **Default**: False

'-a' or '--attack_method' (**Required**)
- **Description**: Specifies the adversarial attack method to use.
- **Options**: 'tf'(TextFooler), 'bae'(BAE)

## Example Command
```python
python text_attack.py -m bert -d yelp -l -a tf
```
This command evaluates the `'bert-base-uncased'` model on the `'yelp_polarity'` dataset using 1,000 randomly sampled test data, indicates that label smoothing was applied during fine-tuning process, and uses the TextFooler adversarial attack method.

# References
[1] [Yang, Y., Dan, S., Roth, D., & Lee, I. (2022). In and out-of-domain text adversarial robustness via label smoothing. arXiv preprint arXiv:2212.10258.](https://arxiv.org/abs/2212.10258)  
[2] [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.](https://arxiv.org/abs/1810.04805)  
[3] [Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.](https://arxiv.org/abs/1910.01108)  
[4] [Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.](https://arxiv.org/abs/1907.11692)  
[5] [Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2019). Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment. CoRR abs/1907.11932 (2019). arXiv preprint arXiv:1907.11932.](https://ojs.aaai.org/index.php/AAAI/article/view/6311)  
[6] [Garg, S., & Ramakrishnan, G. (2020). Bae: Bert-based adversarial examples for text classification. arXiv preprint arXiv:2004.01970.](https://arxiv.org/abs/2004.01970)  