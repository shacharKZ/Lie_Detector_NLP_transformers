import os

'''
given a path to a file, parser it into a string according to a known structure and returns this string
'''
def parser_file(path, file_name) -> str:
    # extract training name (dataset), learning rate, and epochs
    tmp_index = file_name.find('_resInfo')
    train_name = file_name[:tmp_index]
    tmp_index = train_name.find('train')
    train_size = train_name[tmp_index+5:]
    epochs_index = file_name.find("epochs")
    lr_index = file_name.find("_lr")
    epochs = file_name[epochs_index+6:lr_index]
    lr = file_name[lr_index+3:-4]

    # extract the accuracy results for each testing dataset
    sum_acc = 0
    num_of_res = 5
    res = f'\"{file_name}\",\"{train_name}\",{train_size},{epochs},{lr}'
    with open(f'{path}/{file_name}') as f:
        for line in f:
            if 'accuracy result' in line and '_test' in line and '{' not in line:
                tmp_index = line.rfind(" ")
                res_num = float(line[tmp_index:-1])
                sum_acc += res_num
                res += f',{res_num}'
    # also include average result
    avg_acc = sum_acc/num_of_res
    res += f',{avg_acc}'

    # extract which datasets was involved in the training process in boolean columns
    tmp = ''
    sum_domain = 0
    for domain in ['abortaion', 'amazonReviews', 'bestFriend', 'deathPenalty', 'hotels']:
        if domain in file_name:
            sum_domain += 1
            tmp += ',True'
        else:
            tmp += ',False'
    res += f',{sum_domain}{tmp}'
    return res


'''
given a path of a directory containing classified results, collect all those files and parsed each one of them.
the parsed results them output into a csv file (a table with a specific structure)
'''
def parse_all_files(dir_name='./results_all', out_name=None):
    if out_name is None:
        out_name = dir_name
    results_files = os.listdir(dir_name)  # collect all files in the directory
    results_csv = "file_name,train_name,train_size,epochs,lr,abortaion_test40,amazonReviews_test40,bestFriend_test40," \
                  "deathPenalty_test40,hotels_test40,average,Num of domains,include abortion,include amazonReviews," \
                  "include bestFriend,include deathPenalty,include hotels\n"
    for file_name in results_files:  # parsed each file
        curr_res = parser_file(dir_name, file_name)
        results_csv += curr_res + "\n"
    f = open(f'{out_name}_parsed.csv', 'a')  # output all the results into a new csv file
    f.write(results_csv)
    f.close()


if __name__ == '__main__':
    parse_all_files()
    print("DONE")
