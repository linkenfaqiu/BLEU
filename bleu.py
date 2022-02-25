"""
Description:
    1) 使用nltk包中的bleu计算工具来进行辅助计算
"""
import numpy as np
import re
from nltk.translate.bleu_score import corpus_bleu

def my_bleu_v1(candidate_token, reference_token):
    """
    :description:
    最简单的计算方法是看candidate_sentence 中有多少单词出现在参考翻译中, 重复的也需要计算. 
    计算出的数量作为分子，分母是candidate中的单词数量
    :return: 候选句子单词在reference中出现的次数/candidate单词数量
    """
    # 分母是候选句子中单词在参考句子中出现的次数 重复出现也要计算进去
    count = 0
    for token in candidate_token:
        if token in reference_token:
            count += 1
    a = count
    # 计算候选翻译的句子中单词的数量
    b = len(candidate_token)
    return a/b

def calculate_average(precisions, weights):
    """计算几何加权平均值"""
    tmp_res = 1
    for id, item in enumerate(precisions):
        tmp_res = tmp_res*np.power(item, weights[id])
    tmp_res = np.power(tmp_res, np.sum(weights))
    return tmp_res


def calculate_candidate(gram_list, candidate):
    """计算candidate中gram_list的个数。"""
    gram_sub_str = ' '.join(gram_list)
    return len(re.findall(gram_sub_str, candidate))


def calculate_reference(gram_list, references):
    """计算references中gram_list的个数"""
    gram_sub_str = ' '.join(gram_list)
    gram_count = []
    for item in references:
        # 计算子串个数
        gram_count.append(len(re.findall(gram_sub_str, item)))
    return gram_count


def my_bleu_v2(candidate_sentence, reference_sentences, max_gram, weights,mode=0):
    """
    :description: 最初版本的bleu指标存在比较大的缺陷，是以一个词为基准计算分母，现在改进方法采用n个词作为一个组用于计算分母
    其中n可以从1取到最大，这样如果事先决定了所要计算gram的最大长度(N) 那么可以在candidate和reference上计算出每一个
    长度的gram的精度 然后对精度进行几何加权平均即可。
    """
    candidate_corpus = list(candidate_sentence.split(' '))
    # reference的个数
    refer_len = len(reference_sentences)
    candidate_tokens_len = len(candidate_corpus)
    # 首先需要计算各种长度的gram的precision值
    if mode == 0:
        gram_precisions = []
        for i in range(max_gram):
            # 计算每个gram的precision
            # i + 1即为当前的gram
            curr_gram_len = i + 1
            # 计算当前gram的分子长度
            curr_gram_mole = 0
            # 计算当前gram的分母长度
            curr_gram_deno = 0
            for j in range(0, candidate_tokens_len, curr_gram_len):
                if j + curr_gram_len > candidate_tokens_len:
                    continue
                else:
                    curr_gram_list = candidate_corpus[j:j+curr_gram_len]
                    gram_candidate_count = calculate_candidate(curr_gram_list, candidate_sentence)
                    gram_reference_count_list = calculate_reference(curr_gram_list, reference_sentences)
                    truncation_list = []
                    for item in gram_reference_count_list:
                        truncation_list.append(np.min([gram_candidate_count, item]))
                    curr_gram_mole += np.max(truncation_list)
                    curr_gram_deno += gram_candidate_count
            print('current length %d and gram mole %d and deno %d' % (i+1, curr_gram_mole, curr_gram_deno))
            gram_precisions.append(curr_gram_mole/curr_gram_deno)
        print('all the precisions about the grams')
        print(gram_precisions)

        # 第二种计算方法与第一种计算方法本质上的区别在于计算截断计数的区别(最终结果是一样的)
        # 先计算当前n长度的gram在所有的参考文献中的出现次数的最大值 然后在与当前gram在candidate sentence中出现的次数的最小值

    # 其次对多元组合(n-gram)的precision 进行加权取平均作为最终的bleu评估指标
    # 一般选择的做法是计算几何加权平均 exp(sum(w*logP))
        average_res = calculate_average(gram_precisions, weights)
        print('current average result')
        print(average_res)

    # 最后引入短句惩罚项 避免短句翻译结果取得较高的bleu值, 影响到整体评估
    bp = 1
    reference_len_list = [len(item.split(' ')) for item in reference_sentences]
    if candidate_tokens_len in reference_len_list:
        bp = 1
    else:
        if candidate_tokens_len < np.max(reference_len_list):
            bp = np.exp(1-(np.max(reference_len_list)/candidate_tokens_len))
    return bp*average_res

if __name__ == '__main__':
    candidate_sentence = 'hello this is my code'
    reference_sentence = 'hello this code is not mine'
    candidate_token = candidate_sentence.split(' ')
    reference_token = reference_sentence.split(' ')
    bleu_v1_score = my_bleu_v1(candidate_token, reference_token)
    print('bleu version 1 score is %.2f ' % bleu_v1_score)

    predict_sentence = 'how old is the man'
    train_sentences = ['this is a dog and the other is a cat', 'how old are they', 'this is paddlepaddle', 'i work in baidu']
    # weights的代表了1-gram，2-gram，3-gram，4-gram占得比重，缺省情况下为各占0.25。
    bleu_v2_score = my_bleu_v2(predict_sentence, train_sentences, 4, weights=[0.25, 0.25, 0.25, 0.25], mode=0)