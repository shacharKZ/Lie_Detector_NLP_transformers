import time
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score
import datetime
import random
import csv
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = 'roberta-base'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
fix_seed = 42
torch.manual_seed(fix_seed)

OUT_PATH = Path("./data")

now = datetime.datetime.now()
current_time = now.strftime("%H_%M_%S")

# mix train set
all_ds = {
    'bestFriend_train': "./data/bestFriend._v1_train.csv",
    'abortion_train': "./data/abortion._v1_train.csv",
    'deathPenalty_train': "./data/deathPenalty._v1_train.csv",
    # 'reviews_train': "./data/fr_train5555_13.csv",
    # 'reviews_train': "./data/reviews_train7500.csv",
    'reviews_train_3000': "./data/reviews_train3000.csv",
    'hotels_train_v4': "./data/hotels_train_v4.csv",
    # 'politic_v1': "./data/politic-v2.csv",

    #'hotels_dev': "./data/hotel-deceptive-opinion3-dev.csv",
    'hotels_dev': "./data/hotels_dev_v4.csv",
    'reviews_dev': "./data/fr_test5555_13.csv",
    'deathPenalty_dev': "./data/deathPenalty._v1_test.csv",
    'bestFriend_dev': "./data/bestFriend._v1_test.csv",
    'abortion_dev': "./data/abortion._v1_test.csv",

    'mix_hotels_reviews_v2': "./data/mix_train_hotels_reviews_v2.csv",
    'mix_bestFriend_abortion': "./data/bestFriend_and_abortion._v1_train.csv",
    'mix_deathPenalty_bestFriend': "./data/deathPenalty_and_bestFriend._v1_train.csv",
    'mix_deathPenalty_abortion': "./data/deathPenalty_and_abortion._v1_train.csv",
}
raw_ds = load_dataset("csv", data_files=all_ds)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # this is what made the different!

tokenized_ds = raw_ds.map(tokenizer, input_columns='text', fn_kwargs={
    "max_length": 128, "truncation": True, "padding": "max_length"})
tokenized_ds.set_format('torch')
for split in tokenized_ds:
    tokenized_ds[split] = tokenized_ds[split].add_column('label', raw_ds[split]['label'])

print("Will be working with the following datasets:")
for tmp in tokenized_ds:
    print(tmp)


def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    curr_acc = accuracy_score(preds, labels)
    return {'accuracy': curr_acc,
            'eval_accuracy': curr_acc,
            'eval_dev_accuracy': curr_acc}


def build_and_evaluate_model(train_with, max_train_epochs=8, min_train_epochs=3, seed=42, per_device_train_batch_size=10, per_device_eval_batch_size=128,
                             learning_rate=1e-05, weight_decay=0, adam_beta1=0.9, warmup_ratio=0, adafactor=False,
                             skip_bad_results: float = 0, save_score=True):
    if train_with not in tokenized_ds:
        print(f"\n\n<<<<<<<<<< Can't find {train_with} in datasets... >>>>>>>>>>>>>>>>")
        return
    print(f"\n\n<<<<<<<<<< check results for {train_with} >>>>>>>>>>>>>>>>")

    model_clf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    # model_clf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_clf.config.pad_token_id = model_clf.config.eos_token_id

    torch.manual_seed(seed)

    tmp_folder_time = datetime.datetime.now().strftime("%H_%M_%S")
    logs_dir = './logs_' + str(tmp_folder_time)
    os.system(f"mkdir {logs_dir}")
    args = TrainingArguments(output_dir=logs_dir, load_best_model_at_end=False,
                             overwrite_output_dir=True, per_device_train_batch_size=per_device_train_batch_size,
                             per_device_eval_batch_size=per_device_eval_batch_size, evaluation_strategy='epoch',
                             metric_for_best_model='dev_accuracy', greater_is_better=True, do_train=True,
                             num_train_epochs=min_train_epochs, report_to='none', seed=seed, save_strategy='epoch')

    while args.num_train_epochs <= max_train_epochs:
        trainer = Trainer(
            model=model_clf,
            args=args,
            train_dataset=tokenized_ds[train_with],
            eval_dataset=tokenized_ds[train_with],
            compute_metrics=metric_fn,
        )
        if args.num_train_epochs == min_train_epochs:
            trainer.train()
        else:
            ls_res = os.listdir(logs_dir)
            last_index = 0
            last_num = 0
            # print("%%%%%%%%%%%%%%%%%%")
            # print(ls_res)
            for index, val in enumerate(ls_res):
                index_tmp = val.rfind('-')
                curr_num = int(val[index_tmp+1:])
                if curr_num > last_num:
                    last_index = index
                    last_num = curr_num
            last_checkpoint = ls_res[last_index]
            print(f'---- continue from {logs_dir}/{last_checkpoint} ----')
            # print("%%%%%%%%%%%%%%%%%%")
            trainer.train(f'{logs_dir}/{last_checkpoint}')

        # tmp_name = f"{model_name}_tmp_name"
        # model_clf.save_pretrained(str(SAVED_TO_PATH / tmp_name))

        res_dic = {}
        for curr_dev in tokenized_ds:
            if 'train' not in curr_dev:
                curr_res = trainer.predict(tokenized_ds[curr_dev])
                res_dic[curr_dev] = curr_res
                print(f'\nAccuracy score on {curr_dev} is = {curr_res.metrics["test_accuracy"]}')

        file_name = f'./results/{train_with}_resInfo_{str(current_time)}_epochs{args.num_train_epochs}_lr{learning_rate}.txt'
        f = open(file_name, "a")
        for curr_res in res_dic:
            f.write(f'\n\n{curr_res} accuracy result: {res_dic[curr_res].metrics["test_accuracy"]}\n')
            f.write(str(res_dic[curr_res].metrics))

        f.write('\n\n\nThe training args for this run were:')
        f.write(str(args))
        f.close()
        print(f"\nresults can be found at {file_name}")
        args.num_train_epochs += 1

    os.system(f"rm -r {logs_dir}")


if __name__ == '__main__':
    for lr in [1e-5, 3e-5, 5e-5, 6e-5, 7e-5]:
        for ds_name in tokenized_ds:
            if ('dev' not in ds_name) and ('test' not in ds_name):
                build_and_evaluate_model(ds_name, max_train_epochs=9, learning_rate=lr)

    print("DONE")
