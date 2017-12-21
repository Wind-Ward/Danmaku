
import jieba.analyse

jieba.load_userdict("data/metadata/user_dict.txt")
sentence=open("data/processed_danmu/slice_cluster_file/"+"result_33_0.005_2_cluster_slice_0.txt","r").read()
s=jieba.analyse.textrank(sentence,topK=10, withWeight=False)
print(s)
