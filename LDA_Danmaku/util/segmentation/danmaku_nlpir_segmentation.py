import re
import pynlpir
from ctypes import c_char_p
pynlpir.open()

def add_user_dict(file_name="./user_dict.txt"):
    with open(file_name,"r") as f:
        for item in f.readlines():
            #print(item.strip())
            pynlpir.nlpir.AddUserWord(c_char_p(item.strip().encode()))

def add_stop_words(file_name="./stopWords.txt"):
    stop_words=[]
    with open(file_name,"r") as f:
        for item in f.readlines():
            stop_words.append(item.strip())
    return set(stop_words)


def segment(pos_tag,read_file_name="./origin_33.txt",write_file_name="./segmented_nlpir_33.txt"):
    result = []
    stop_words=add_stop_words()
    with open(read_file_name,"r") as f:
        for item in f.readlines():
            temp=pynlpir.segment(item.strip())
            _=[]
            for word in temp:
                if word[1] not in pos_tag and word[0] not in stop_words:
                    pattern = re.compile("[a-zA-Z0-9@]+")
                    m = pattern.match(word[0])
                    if not m and len(word[0])>1:
                        _.append(word[0])
            result.append(_)


    with open(write_file_name,"w") as f:
        for item in result:
            if len(item)>=3:
                f.write(" ".join(item))
                f.write("\n")


if __name__ == '__main__':
    add_user_dict()
    pos_tag=["conjunction","punctuation mark","adverb","modal particle",\
             "preposition","numeral","classifier","interjection","particle","onomatopoeia",None]
    segment(pos_tag)