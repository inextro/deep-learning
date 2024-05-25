import os
import argparse
import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='[bert, dbert]')
    parser.add_argument('-d', '--data_name', type=str, required=True, help='[yelp, sst2, ag_news, movie_review]')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)

    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    num_samples = args.num_samples


    # 데이터 불러오기
    if data_name == 'yelp':
        data = load_dataset('yelp_polarity', trust_remote_code=True)
        test_data = data['test'].shuffle(seed=42).select(range(num_samples)) # 전체 평가 데이터 중 1,000개를 무작위로 선택
    elif data_name == 'sst2':
        data = load_dataset('glue', 'sst2', trust_remote_code=True)
        test_data = data['test'].shuffle(seed=42).select(range(num_samples))
    elif data_name == 'ag_news':
        data = load_dataset('ag_news', trust_remote_code=True)
        test_data = data['test'].shuffle(seed=42).select(range(num_samples))
    elif data_name == 'movie_review':
        raise NotImplementedError('movie_review dataset is not implemented')
    else:
        raise ValueError('Unknown data name')


    # 모델 불러오기
    num_labels = len(set(test_data['label']))
    model_save_dir = './result/saved_model'

    if model_name == 'bert':
        model_path = os.path.join(model_save_dir, model_name + '_' + data_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'dbert':
        model_path = os.path.join(model_save_dir, model_name + '_' + data_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('distil-bert-uncased')
    else:
        raise ValueError('Unknown model name')
    

    # tokenize function 정의
    def tokenize_function(examples):
        if data_name in ['yelp', 'ag_news']:
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        elif data_name == 'sst2':
            return tokenizer(examples['sentence'], padding='max_length', truncation=True)
        elif data_name == 'movie_review':
            raise NotImplementedError('movie_review dataset is not implemented')

    tokenized_data = test_data.map(tokenize_function, batched=True, batch_size=64)


    # 성능 평가
    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir = './results', 
        fp16=True # mixed precision training
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        eval_dataset=tokenized_data, 
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()
    print(f"Accuracy: {results['eval_accuracy']: .4f}")
    

if __name__ == '__main__':
    main()