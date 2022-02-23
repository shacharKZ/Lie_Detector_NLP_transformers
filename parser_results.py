import os


def parser_file(path, file_name) -> str:
    # print(f'{path}/{file_name}')
    tmp_index = file_name.find('_resInfo')
    train_name = file_name[:tmp_index]
    tmp_index = train_name.find('train')
    train_size = train_name[tmp_index+5:]
    epochs_index = file_name.find("epochs")
    lr_index = file_name.find("_lr")
    epochs = file_name[epochs_index+6:lr_index]
    lr = file_name[lr_index+3:-4]
    res = f'\"{file_name}\",\"{train_name}\",{train_size},{epochs},{lr}'
    with open(f'{path}/{file_name}') as f:
        for line in f:
            if 'accuracy result' in line and '_test' in line and '{' not in line:
                tmp_index = line.rfind(" ")
                res_num = float(line[tmp_index:-1])
                # print(res_num)
                res += f',{res_num}'
    return res


def parser_all_files(dir_name='./results220222'):
    results_files = os.listdir(dir_name)
    results_csv = "file_name,trian_name,train_size,epochs,lr,abortaion_test40,amazonReviews_test40,bestFriend_test40,deathPenalty_test40,hotels_test40\n"
    for file_name in results_files:
        curr_res = parser_file(dir_name, file_name)
        # print(curr_res)
        results_csv += curr_res + "\n"
    # print(results_csv)
    f = open(f'{dir_name}_parsed.csv', 'a')
    f.write(results_csv)
    f.close()


if __name__ == '__main__':
    parser_all_files()
    print("DONE")
