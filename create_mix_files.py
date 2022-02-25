import os
import itertools

'''
given a path to a directory of training set files (csv), creates all the possible training set containing n 
domain (files). each one of the new training set should have an equal number of samples of each domain it contain.
'''
def create_mix_files_from_n_domains(dir_name='./datasets v2', out_dir='./datasets v4', n=5):
    results_files = os.listdir(dir_name)
    print(results_files)
    for curr_set in itertools.combinations(results_files, n):
        flag_stop = False
        for domain in curr_set:
            if 'csv' not in domain or 'train' not in domain:
                flag_stop = True
                continue

        if flag_stop:
            continue

        size1_index = curr_set[0].rfind('_train')
        size1 = curr_set[0][size1_index:]
        _num = curr_set[0].count('_')
        new_file_name = 'mix' + str(n) + '_'
        for domain in curr_set:
            size2_index = domain.rfind('_train')
            size2 = domain[size2_index:]
            if size2 != size1:
                flag_stop = True
                continue
            if domain.count('_') != _num or domain.count('_') != 1:
                flag_stop = True
                continue
            new_file_name += domain[:size2_index] + '_'

        if flag_stop:
            continue
        new_size = int(size1[6:-4])*n
        new_file_name += 'train' + str(new_size) + ".csv"
        print(new_file_name, '<--', curr_set)
        f_new = open(f'{out_dir}/{new_file_name}', 'a+')
        for index, domain in enumerate(curr_set):
            f1 = open(f'{dir_name}/{domain}', 'r')
            if index != 0:
                f1.readline()
            f_new.write(f1.read())
            f1.close()
        f_new.close()


if __name__ == '__main__':
    create_mix_files_from_n_domains(n=2)
    create_mix_files_from_n_domains(n=3)
    create_mix_files_from_n_domains(n=4)
    create_mix_files_from_n_domains(n=5)
    print("DONE")
