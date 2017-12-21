import re
import jieba
import jieba.posseg as pseg
import copy
import os


def address_file(file_dir,des_dir):
    file_list=os.listdir(file_dir)
    for file in file_list:
        file_path=os.path.join(file_dir,file)
        temp=[]
        with open(file_path,"r") as f:
            lines=f.readlines()
            for line in lines:
                pattern = re.compile("^<d p=\"(.+)\">(.+)</d>")
                m = pattern.match(line.strip())
                if m:
                    temp.append({"time":int(float(m.group(1).split(',')[0])),"content":m.group(2)})
        lines = sorted(temp, key=lambda e: (e.__getitem__('time')))

        with open(os.path.join(des_dir,file),"w") as f2:
            for item in lines:
                f.write(item["content"]+"\n")




if __name__ == '__main__':
    address_file()