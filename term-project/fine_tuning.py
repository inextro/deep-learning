import os
import argparse
import evaluate
import numpy as np

from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='[bert, dbert]')
    parser.add_argument('-d', '--data_name', type=str, required=True, help='[yelp, sst2, ag_news, movie_review]')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    batch_size = args.batch_size


    # 데이터 불러오기
    if data_name == 'yelp':
        data = load_dataset('yelp_polarity', trust_remote_code=True)
        train_data = data['train']
    elif data_name == 'sst2':
        data = load_dataset('glue', 'sst2', trust_remote_code=True)
        train_data = data['train']
    elif data_name == 'ag_news':
        data = load_dataset('ag_news', trust_remote_code=True)
        train_data = data['train']
    elif data_name == 'movie_review':
        pass
    else:
        raise ValueError('Unknown data name')


    # 모델 불러오기
    num_labels = len(set(data['train']['label']))

    if model_name == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'dbert':
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        raise ValueError('Unknown model name')


    # fine-tuning
    def tokenize_function(examples):
        if data_name in ['yelp', 'ag_news']:
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        elif data_name == 'sst2':
            return tokenizer(examples['sentence'], padding='max_length', truncation=True)
        elif data_name == 'movie_review':
            return None
    
    tokenized_data = train_data.map(tokenize_function, batched=True, batch_size=batch_size)
    # small_tokenized_data = tokenized_data.shuffle(seed=42).select(range(1000)) # 전체 학습 데이터 중 1000개만 무작위로 선택

    output_dir = './output'
    if not os.path.exists(output_dir): # 저장경로가 존재하지 않으면 해당 경로 생성
        os.makedirs(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir, 
        eval_strategy='epoch', # epoch가 끝날 때 마다 accuracy 확인
        num_train_epochs=3, 
        learning_rate = 5e-6, 
        fp16=True # mixed precision training
    )

    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_data, 
        # train_dataset=small_tokenized_data, 
        compute_metrics=compute_metrics
    )
    trainer.train()

    # fine-tuned 모델 저장
    model_save_dir = './result/saved_model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir) # 저장경로가 존재하지 않으면 해당 경로 생성
    model.save_pretrained(os.path.join(model_save_dir, model_name + '_' +data_name))


if __name__ == '__main__':    
    main()