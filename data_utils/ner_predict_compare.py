# 模型预测结果和标注结果进行比较输出
import pandas as pd


def main(predict_txt: str, test_txt: str):
    """

    :param predict_txt: 模型预测结果，文件一般位于模型输出目录predicti.txt文件
    :param test_txt: 已标注的测试数据
    :return:
    """
    predict_tags = []
    with open(predict_txt, 'r', encoding='utf-8') as f:
        one = []
        for line in f:
            one = line.strip().split(' ')
            predict_tags.append(one)

    rows = []
    with open(test_txt, 'r', encoding='utf-8') as f:
        sentence = []
        true_tags = []
        i = 0
        for line in f:
            if line.strip() == '':
                one_predict_tags = predict_tags[i]
                for char, true_tag, predict_tag in zip(sentence, true_tags, one_predict_tags):
                    rows.append([char, true_tag, predict_tag])
                i += 1
                sentence = []
                true_tags = []
                continue
            char, true_tag = line.strip('\n').split()
            sentence.append(char)
            true_tags.append(true_tag)

    df = pd.DataFrame.from_records(data=rows, columns=['字', '原始', '预测'])
    df.to_excel('比较结果.xls')


if __name__ == "__main__":
    main('prediction.txt', 'test.txt')
