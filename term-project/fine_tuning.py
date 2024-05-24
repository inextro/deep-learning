import os
import argparse

from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='[bert, dbert]')
    parser.add_argument('-d', '--data_name', type=str, required=True, help='[yelp, sst2, ag_news, movie_review]')
    
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name


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
        if data_name == 'yelp' or data_name == 'ag_news':
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        elif data_name == 'sst2':
            return tokenizer(examples['sentence'], padding='max_length', truncation=True)
        elif data_name == 'movie_review':
            return None
        
    training_args = TrainingArguments(
        output_dir='./output',
        num_train_epochs=3
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_data.map(tokenize_function, batched=True), 
        optimizers=(AdamW(model.parameters(), lr=5e-6), None) # None: lr-scheduling 사용하지 않음
    )
    trainer.train()

    # fine-tuned 모델 저장
    model_save_dir = './result/saved_model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir) # 저장경로가 존재하지 않으면 해당 경로 생성
    model.save_pretrained(os.path.join(model_save_dir, model_name + '_' +data_name))


if __name__ == '__main__':    
    main()