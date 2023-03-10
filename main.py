# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from bert4torch.tokenizers import Tokenizer
import config
import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
def datalabel():
    path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/data/medical.train'
    D = []
    flags = []
    with open(path, encoding='utf-8') as f:
        f = f.readlines()
        for i in f:
            if i != '\n':
                char, flag = i.split(' ')
                flags.append(flag)
            else:
                continue
        flags_ = set(flags)
        print(flags_)

def load_data():
    file_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/data/medical.train'
    D = []
    with open(file_path, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D
def collate_fn():
    batch_token_ids, batch_labels = [], []
    d =  ['现头昏口苦', [3, 4, '临床表现']]
    tokenizer = Tokenizer('D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/pre_trained_model/vocab.txt',
                          do_lower_case=True)
    tokens = tokenizer.tokenize(d[0], maxlen=128)
    mapping = tokenizer.rematch(d[0], tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    token_ids = tokenizer.tokens_to_ids(tokens)
    labels = np.zeros(len(token_ids))
    for start, end, label in d[1:]:
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]
            labels[start] = config.categories_label2id['B-' + label]
            labels[start + 1:end + 1] = config.categories_label2id['I-' + label]
    batch_token_ids.append(token_ids)
    batch_labels.append(labels)
    print(batch_labels)
    print(batch_token_ids)
def categories():
    categories = ['O', 'B-中医治疗', 'I-中医治疗', 'B-方剂', 'I-方剂', 'B-中医诊断', 'I-中医诊断', 'B-中医证候',
                  'I-中医证候', 'B-西医诊断',
                  'I-西医诊断', 'B-其他治疗', 'I-其他治疗', 'B-中药', 'I-中药', 'B-临床表现', 'I-临床表现',
                  'B-西医治疗', 'I-西医治疗',
                  'B-中医治则', 'I-中医治则']

    categories_label2id = {k: i for i, k in enumerate(categories)}
    categories_id2label = {i: k for i, k in enumerate(categories)}
    print(categories_id2label)
    print(categories_label2id)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    collate_fn()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
