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

# model_name = 'gpt2'
model_name = 'roberta-base'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
fix_seed = 42
torch.manual_seed(fix_seed)

OUT_PATH = Path("./data")

now = datetime.datetime.now()
current_time = now.strftime("%H_%M_%S")

# mix train set
# all_ds = {
#     'abortaion_test40': './datasets v2/abortaion_test40.csv',
#     'abortaion_train160': './datasets v2/abortaion_train160.csv',
#     'amazonReviews_test40': './datasets v2/amazonReviews_test40.csv',
#     'amazonReviews_train1000': './datasets v2/amazonReviews_train1000.csv',
#     'amazonReviews_train160': './datasets v2/amazonReviews_train160.csv',
#     'bestFriend_test40': './datasets v2/bestFriend_test40.csv',
#     'bestFriend_train160': './datasets v2/bestFriend_train160.csv',
#     'deathPenalty_test40': './datasets v2/deathPenalty_test40.csv',
#     'deathPenalty_train160': './datasets v2/deathPenalty_train160.csv',
#     'hotels_test40': './datasets v2/hotels_test40.csv',
#     'hotels_train1000': './datasets v2/hotels_train1000.csv',
#     'hotels_train160': './datasets v2/hotels_train160.csv',
#
#     'mix_abortaion_bestFriend_train160': './datasets v2/abortaion_bestFriend_train160.csv',
#     'mix_abortaion_bestFriend_train320': './datasets v2/abortaion_bestFriend_train320.csv',
#     'mix_abortaion_deathPenalty_train160': './datasets v2/abortaion_deathPenalty_train160.csv',
#     'mix_abortaion_deathPenalty_train320': './datasets v2/abortaion_deathPenalty_train320.csv',
#     'mix_bestFriend_deathPenalty_train160': './datasets v2/bestFriend_deathPenalty_train160.csv',
#     'mix_bestFriend_deathPenalty_train320': './datasets v2/bestFriend_deathPenalty_train320.csv',
#     'mix_bestFriend_hotels_train160': './datasets v2/bestFriend_hotels_train160.csv',
#     'mix_bestFriend_hotels_train320': './datasets v2/bestFriend_hotels_train320.csv',
#     'mix_amazonReviews_abortaion_train160': './datasets v2/amazonReviews_abortaion_train160.csv',
#     'mix_amazonReviews_bestFriend_train160': './datasets v2/amazonReviews_bestFriend_train160.csv',
#     'mix_amazonReviews_hotels_train160': './datasets v2/amazonReviews_hotels_train160.csv',
#     'mix_amazonReviews_hotels_train320': './datasets v2/amazonReviews_hotels_train320.csv',
# }

all_ds = {
    'abortaion_test40': './datasets v2/abortaion_test40.csv',
    'amazonReviews_test40': './datasets v2/amazonReviews_test40.csv',
    'bestFriend_test40': './datasets v2/bestFriend_test40.csv',
    'deathPenalty_test40': './datasets v2/deathPenalty_test40.csv',
    'hotels_test40': './datasets v2/hotels_test40.csv',

    # 'abortaion_train120': './datasets v3/abortaion_train120.csv',
    # 'abortaion_train40': './datasets v3/abortaion_train40.csv',
    # 'abortaion_train80': './datasets v3/abortaion_train80.csv',
    # 'amazonReviews_train120': './datasets v3/amazonReviews_train120.csv',
    # 'amazonReviews_train40': './datasets v3/amazonReviews_train40.csv',
    # 'amazonReviews_train80': './datasets v3/amazonReviews_train80.csv',
    # 'bestFriend_train120': './datasets v3/bestFriend_train120.csv',
    # 'bestFriend_train40': './datasets v3/bestFriend_train40.csv',
    # 'bestFriend_train80': './datasets v3/bestFriend_train80.csv',
    # 'deathPenalty_train120': './datasets v3/deathPenalty_train120.csv',
    # 'deathPenalty_train40': './datasets v3/deathPenalty_train40.csv',
    # 'deathPenalty_train80': './datasets v3/deathPenalty_train80.csv',
    # 'hotels_train120': './datasets v3/hotels_train120.csv',
    # 'hotels_train40': './datasets v3/hotels_train40.csv',
    # 'hotels_train80': './datasets v3/hotels_train80.csv',
    # 'mix_abortaion_amazonReviews_train160': './datasets v3/mix_abortaion_amazonReviews160.csv',
    # 'mix_abortaion_amazonReviews_train240': './datasets v3/mix_abortaion_amazonReviews240.csv',
    # 'mix_abortaion_amazonReviews_train320': './datasets v3/mix_abortaion_amazonReviews320.csv',
    # 'mix_abortaion_amazonReviews_train80': './datasets v3/mix_abortaion_amazonReviews80.csv',
    # 'mix_abortaion_bestFriend_train160': './datasets v3/mix_abortaion_bestFriend160.csv',
    # 'mix_abortaion_bestFriend_train240': './datasets v3/mix_abortaion_bestFriend240.csv',
    # 'mix_abortaion_bestFriend_train320': './datasets v3/mix_abortaion_bestFriend320.csv',
    # 'mix_abortaion_bestFriend_train80': './datasets v3/mix_abortaion_bestFriend80.csv',
    # 'mix_abortaion_deathPenalty_train160': './datasets v3/mix_abortaion_deathPenalty160.csv',
    # 'mix_abortaion_deathPenalty_train240': './datasets v3/mix_abortaion_deathPenalty240.csv',
    # 'mix_abortaion_deathPenalty_train320': './datasets v3/mix_abortaion_deathPenalty320.csv',
    # 'mix_abortaion_deathPenalty_train80': './datasets v3/mix_abortaion_deathPenalty80.csv',
    # 'mix_abortaion_hotels_train160': './datasets v3/mix_abortaion_hotels160.csv',
    # 'mix_abortaion_hotels_train240': './datasets v3/mix_abortaion_hotels240.csv',
    # 'mix_abortaion_hotels_train320': './datasets v3/mix_abortaion_hotels320.csv',
    # 'mix_abortaion_hotels_train80': './datasets v3/mix_abortaion_hotels80.csv',
    # 'mix_amazonReviews_bestFriend_train160': './datasets v3/mix_amazonReviews_bestFriend160.csv',
    # 'mix_amazonReviews_bestFriend_train240': './datasets v3/mix_amazonReviews_bestFriend240.csv',
    # 'mix_amazonReviews_bestFriend_train320': './datasets v3/mix_amazonReviews_bestFriend320.csv',
    # 'mix_amazonReviews_bestFriend_train80': './datasets v3/mix_amazonReviews_bestFriend80.csv',
    # 'mix_amazonReviews_deathPenalty_train160': './datasets v3/mix_amazonReviews_deathPenalty160.csv',
    # 'mix_amazonReviews_deathPenalty_train240': './datasets v3/mix_amazonReviews_deathPenalty240.csv',
    # 'mix_amazonReviews_deathPenalty_train320': './datasets v3/mix_amazonReviews_deathPenalty320.csv',
    # 'mix_amazonReviews_deathPenalty_train80': './datasets v3/mix_amazonReviews_deathPenalty80.csv',
    # 'mix_amazonReviews_hotels_train160': './datasets v3/mix_amazonReviews_hotels160.csv',
    # 'mix_amazonReviews_hotels_train2000': './datasets v3/mix_amazonReviews_hotels2000.csv',
    # 'mix_amazonReviews_hotels_train240': './datasets v3/mix_amazonReviews_hotels240.csv',
    # 'mix_amazonReviews_hotels_train320': './datasets v3/mix_amazonReviews_hotels320.csv',
    # 'mix_amazonReviews_hotels_train80': './datasets v3/mix_amazonReviews_hotels80.csv',
    # 'mix_bestFriend_deathPenalty_train160': './datasets v3/mix_bestFriend_deathPenalty160.csv',
    # 'mix_bestFriend_deathPenalty_train240': './datasets v3/mix_bestFriend_deathPenalty240.csv',
    # 'mix_bestFriend_deathPenalty_train320': './datasets v3/mix_bestFriend_deathPenalty320.csv',
    # 'mix_bestFriend_deathPenalty_train80': './datasets v3/mix_bestFriend_deathPenalty80.csv',
    # 'mix_bestFriend_hotels_train160': './datasets v3/mix_bestFriend_hotels160.csv',
    # 'mix_bestFriend_hotels_train240': './datasets v3/mix_bestFriend_hotels240.csv',
    # 'mix_bestFriend_hotels_train320': './datasets v3/mix_bestFriend_hotels320.csv',
    # 'mix_bestFriend_hotels_train80': './datasets v3/mix_bestFriend_hotels80.csv',
    # 'mix_deathPenalty_hotels_train160': './datasets v3/mix_deathPenalty_hotels160.csv',
    # 'mix_deathPenalty_hotels_train240': './datasets v3/mix_deathPenalty_hotels240.csv',
    # 'mix_deathPenalty_hotels_train320': './datasets v3/mix_deathPenalty_hotels320.csv',
    # 'mix_deathPenalty_hotels_train80': './datasets v3/mix_deathPenalty_hotels80.csv'

    'mix3_amazonReviews_bestFriend_hotels_train120': './datasets v4/mix3_amazonReviews_bestFriend_hotels_train120.csv',
    'mix3_amazonReviews_bestFriend_hotels_train240': './datasets v4/mix3_amazonReviews_bestFriend_hotels_train240.csv',
    'mix3_amazonReviews_bestFriend_hotels_train360': './datasets v4/mix3_amazonReviews_bestFriend_hotels_train360.csv',
    'mix3_amazonReviews_bestFriend_hotels_train480': './datasets v4/mix3_amazonReviews_bestFriend_hotels_train480.csv',

    'mix3_abortaion_bestFriend_deathPenalty_train120': './datasets v4/mix3_abortaion_bestFriend_deathPenalty_train120.csv',
    'mix3_abortaion_bestFriend_deathPenalty_train240': './datasets v4/mix3_abortaion_bestFriend_deathPenalty_train240.csv',
    'mix3_abortaion_bestFriend_deathPenalty_train360': './datasets v4/mix3_abortaion_bestFriend_deathPenalty_train360.csv',
    'mix3_abortaion_bestFriend_deathPenalty_train480': './datasets v4/mix3_abortaion_bestFriend_deathPenalty_train480.csv',

    # 'mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train160': './datasets v4/mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train160.csv',
    # 'mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train320': './datasets v4/mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train320.csv',
    # 'mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train480': './datasets v4/mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train480.csv',
    # 'mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train640': './datasets v4/mix4_abortaion_amazonReviews_bestFriend_deathPenalty_train640.csv',

    'mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train200': './datasets v4/mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train200.csv',
    'mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train400': './datasets v4/mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train400.csv',
    'mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train600': './datasets v4/mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train600.csv',
    'mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train800': './datasets v4/mix5_abortaion_amazonReviews_bestFriend_deathPenalty_hotels_train800.csv',




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


def build_and_evaluate_model(train_with, max_train_epochs=12, min_train_epochs=2, seed=42, per_device_train_batch_size=10,
                             per_device_eval_batch_size=128, learning_rate=1e-05, weight_decay=0, adam_beta1=0.9,
                             warmup_ratio=0, adafactor=False, skip_bad_results: float = 0, save_score=True):
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
            if 'test' in curr_dev:
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
    for lr in [1e-5, 3e-5, 5e-5, 7e-5]:
        for ds_name in tokenized_ds:
            if 'train' in ds_name:
                build_and_evaluate_model(ds_name, max_train_epochs=12, learning_rate=lr)

    print("DONE")
