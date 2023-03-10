'''
配置文件
'''

maxlen = 128
batch_size = 16

config_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/mc_bert/bert_config.json'
checkpoint_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/mc_bert/pytorch_model.bin'
dict_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/mc_bert/vocab.txt'
train_data_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/data/medical.train'
vaild_data_path = 'D:/Users/ShuaiqiDuan/Desktop/Research/DL/NLP_Pytorch/medical_ner/data/medical.dev'

categories = ['O','B-中医治疗','I-中医治疗','B-方剂','I-方剂','B-中医诊断','I-中医诊断','B-中医证候','I-中医证候','B-西医诊断',
              'I-西医诊断','B-其他治疗','I-其他治疗','B-中药','I-中药','B-临床表现','I-临床表现','B-西医治疗','I-西医治疗',
              'B-中医治则','I-中医治则']

categories_label2id = {k: i for i,k in enumerate(categories)}
categories_id2label = {i:k for i,k in enumerate(categories)}

