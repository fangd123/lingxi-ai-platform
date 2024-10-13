# @Time : 2020/11/19 3:36 下午
# @Author : Gao Yudong
# @Email   : gaoyudong0628@163.com
# @Software: PyCharm
# @Project: novel_to_script
# @FileName: deal

from pathlib import Path
import pandas as pd
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
from pathlib import Path
import random, re
from config.setting import RE_DIALOG


def save_content(path, content):
    with open(path, "w") as w:
        w.write("\n".join(content))


def read_txt(path):
    with open(path, "r") as r:
        return r.readlines()


def get_content(file):
    """
    通过文件名获取文件内容
    """
    try:
        with open(str(file), "r", encoding="gbk") as r:
            lines = r.readlines()
    except:
        try:
            with open(str(file), "r", encoding="utf-8") as r:
                lines = r.readlines()
        except:
            print('question:', str(file))
            return False
    return lines


def create_train_txt(path, name_list, save_path, column=None):
    """
    创建可训练的数据
    :param path: tsv文件路径
    :param name_list: 文件名列表
    :param column: 标签名
        默认所有标签，单独过滤某一类标签直接指定该标签即可
    :return:
        生成name_list相同文件名的文件
    """

    for name in name_list:
        with open(path + name, "r") as r:
            contents = r.readlines()

        last = []
        for i in contents[1:]:
            if len(i.strip()) == 0: continue
            if "-DOCSTART" in i:
                last.append("\n")
            else:
                split_line = i.split("\t")
                label = split_line[-1]
                if column:
                    if not ("O" in label or column in label):
                        split_line[-1] = "O\n"
                line = " ".join(split_line)
                last.append(line)

        with open(save_path + name.split(".")[0] + ".txt", "w") as w:
            w.writelines(last)


def create_text_classify_test(text_classify_lines, is_save=False):
    """

    """
    def get_dialogs(content):
        """
        获取对话内容，即引号及其内部的文字
        """
        dialogs = re.findall(RE_DIALOG, content)
        return dialogs
    col_index = 0
    text_classify_df_data = []
    for line in text_classify_lines:
        dialogs = get_dialogs(line)
        sentence2 = "$".join(dialogs) if len(dialogs) >= 1 else dialogs[0]
        text_classify_df_data.append([col_index, line.strip(), sentence2.strip()])
        col_index += 1
    df = pd.DataFrame(text_classify_df_data, columns=["index", "sentence1", "sentence2"])
    if is_save: df.to_csv("test.tsv", sep="\t", index=False)
    return df


def create_predict_document(path, save_pathname):
    """
    生成标签都为O的预测文件
    :param path:
    :return:
    """
    files = Path(path).glob('*.txt')
    all_ = []
    for f in files:
        with open(str(f), 'r') as r:
            content = r.readlines()
        for sentence in content:
            for char in sentence.strip():
                all_.append(char + " O" + "\n")
            all_.append("\n")

    with open(save_pathname, "w") as w:
        w.writelines(all_)


def create_predict_document_by_content(content, save_pathname):
    """
    生成标签都为O的预测文件
    :param path:
    :return:
    """
    all_ = []
    for sentence in content:
        for char in sentence.strip():
            all_.append(char + " O" + "\n")
        all_.append("\n")

    with open(save_pathname, "w") as w:
        w.writelines(all_)


def create_predict_content_by_content(content):
    """
    生成标签都为O的预测文件
    :param path:
    :return:
    """
    all_ = []
    for sentence in content:
        for char in sentence.strip():
            all_.append(char + " O" + "\n")
        all_.append("\n")
    return all_


def jsonl2tsv(from_path, to_path, ratio=[6, 3, -1]):
    def tokenizer(sentence: str) -> list:
        return [x for x in sentence]

    files = Path(from_path).glob('*.jsonl')

    sentences = []
    for file in files:
        dataset = read_jsonl(filepath=str(file), dataset=NERDataset, encoding='utf-8')
        conll = dataset.to_conll2003(tokenizer=tokenizer)
        sentences.extend([x['data'].replace(' _ _ ', '\t') for x in conll])

    random.shuffle(sentences)

    if ratio[-1] == -1:
        first_num = int(len(sentences) * ratio[0] / sum(ratio[:-1]))
        with open(to_path + 'train.tsv', 'w', encoding='utf-8') as train, \
                open(to_path + 'dev.tsv', 'w') as dev, \
                open(to_path + 'test.tsv', 'w') as test:
            train.writelines(sentences[:first_num])
            dev.writelines(sentences[first_num:])
            test.writelines(sentences[:])
    else:
        first_num = int(len(sentences) * ratio[0] / sum(ratio))
        two_num = int(len(sentences) * sum(ratio[:-1]) / sum(ratio))
        with open(to_path + 'train.tsv', 'w', encoding='utf-8') as train, \
                open(to_path + 'dev.tsv', 'w') as dev, \
                open(to_path + 'test.tsv', 'w') as test:
            train.writelines(sentences[:first_num])
            dev.writelines(sentences[first_num:two_num])
            test.writelines(sentences[two_num:])
