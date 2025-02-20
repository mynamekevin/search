import jieba
import jieba.posseg
import jieba.analyse
import re
import string
import math
import time
import numpy as np

data_path = "../data/qa_data/"

%matplotlib inline
import matplotlib.pyplot as plt 

date = time.strftime("%Y%m%d", time.localtime())

def load_data(path, gap = "", cols = [0], skip_head = False, max_num = -1):
    output = []
    with open(path, 'r') as f:
        num = 0
        for line in f:
            # 跳过首行
            if skip_head and num == 0:
                num += 1
                continue
                
            line = line.strip()
            if line == "":
                continue
                
            if gap != "":
                sp = line.split(gap)
            else:
                sp = [line]
            if len(sp) < len(cols) or len(sp) <= max(cols):
                continue
                
            line_sp = []
            for col in cols:
                line_sp.append(sp[col])

            if len(line_sp) == 1:
                output.append(line_sp[0])
            else:
                output.append(line_sp)
                
            if max_num != -1 and num > max_num:
                break
            num += 1
    return output

def cut(texts, user_dict = None):
    # 切词
    texts_segs = {}
    if user_dict:
        jieba.load_userdict(user_dict)

    for text in texts:
        segs = jieba.cut(text, cut_all=False)
        # 去掉空值term
        segs = [term for term in list(segs) if term.strip() != "" ]
        texts_segs[text] = segs
    return texts_segs

def query_prepare(data):
    pattern = "[^0-9^A-Z^a-z^\u4e00-\u9fa5]" # 匹配非中文字母数字
    
    query_title_click = {}
    for line in data:
        if len(line) < 3:
            continue
        # query 预处理
        query = line[0].lower()
        query = re.sub(pattern, " ", query)
        if query.strip() == "":
            continue
        # title 预处理
        title = line[1].lower()
        try:
            click = int(line[2]) # click异常数据处理
        except:
            click = 1
            
        if query not in query_title_click:
            query_title_click[query] = {}
            query_title_click[query]["_click"] = 0

        if title not in query_title_click[query]:
            query_title_click[query][title] = 0
        
        query_title_click[query]["_click"] += click 
        query_title_click[query][title] += click
    
    # 平滑处理，防止click过高
    for query, info in query_title_click.items():
        query_title_click[query]["_click_new"] = int(info["_click"] / 100 + 1)
        
    return query_title_click

def smooth_prepare(x, segs, stop_words, init_weight_chaju):
    x_min = min(x)
    x_max = max(x)
    # 缩放到1-2之间，但是对于imp初始权重来说差距太小，极端情况下也就0.3和0.7的差距
    # xs = [round((a - x_min)/(x_max - x_min + 0.0001) + 1, 4) for a in x]
    # 缩放
    xs = [a / x_min for a in x]
    
    p8 = np.percentile(xs, 8)
    p25 = np.percentile(xs, 25)
    p75 = np.percentile(xs, 75)
    xs_mean = np.mean(xs)
    
    y = []
    for w, seg in zip(xs, segs):
        # 单字停用词
        if seg in stop_words and len(seg) == 1:
            w = min(p8, w)
        # 停用词 或者 单字
        elif seg in stop_words or len(seg) == 1:
            w = min(p25, w)
        else:
            w = w
        # 再通过log归一化
        w = math.log(w, init_weight_chaju) + 1
        y.append(round(w, 4))
    
    # test = [round(a/sum(y), 3) for a in y]
    # print("tf:", x, "zo:", xs, "log:", y, "w:", test, "\n")
    return y
    
def imp_init_weight(query_title_click, query_segs, stop_words, init_weight_chaju):
    query_termWeight = {}  # key=query, value=[1,3,1,10] 和 query_segs的value列表一一对应
    # n = 0
    for query, title_click_map in query_title_click.items():
        segs = query_segs[query]
        query_termWeight[query] = [1] * len(segs)
        
        for i, seg in enumerate(segs):
            for title, click in title_click_map.items():
                if title == "_click" or title == "_click_new":
                    continue
                if title.find(seg) >= 0:
                    query_termWeight[query][i] += click
        # ctf 平滑处理
        # tmp = query_termWeight[query]
        query_termWeight[query] = smooth_prepare(query_termWeight[query], segs, stop_words, init_weight_chaju)
        
        # print(query, segs, tmp, query_termWeight[query])
        # if n > 100:
        #     break
        # n += 1
    return query_termWeight

def imp(text_segs, query_title_click, text_termWeight, loops = 2, max_term_num = -1):
    term_imp_list = {}
    term_new_imp = {}
    # 循环次数
    for loop in range(loops):
        # 计算query中每个term的权重占比， T_mpi = Bt / sum(Bti), Bt是term权重
        for text, segs in text_segs.items():
            if max_term_num > 0 and len(segs) > max_term_num:
                segs = segs[:max_term_num]
            # 找到每个term的权重 Bt
            segs_weight_list = []
            # 初始权重
            if loop == 0:
                segs_weight_list = text_termWeight[text]
            else: 
                for seg in segs:
                    try:
                        seg_weight = term_new_imp[seg][0]
                    except:
                        # print(seg, i, len(term_new_imp))
                        continue
                    segs_weight_list.append(seg_weight)
            segs_wegiht_sum = sum(segs_weight_list)
            # 计算 T_mpi，保存在term_imp_list中
            for (seg, weight) in zip(segs, segs_weight_list):
                new_weight = weight / segs_wegiht_sum
                if seg not in term_imp_list:
                    term_imp_list[seg] = []
                # 点击n次，这里要加n个new_weight
                click_new = min(50000, query_title_click[text]["_click_new"])
                term_imp_list[seg].extend([new_weight] * click_new) 

        # 重新计算term权重  Bt = N / sum(1/Tmpi), N是包含第i个term的query数量
        for (seg, weight_list) in term_imp_list.items():
            weight_list_tmp = []
            for weight in weight_list:
                weight_list_tmp.append(1 / weight)
            # 统计词频，用于判断term权重置信度
            term_new_imp[seg] = [len(weight_list) / sum(weight_list_tmp), len(weight_list)]
            # 算完后，要clear 便于下一轮计算
            term_imp_list[seg] = []
    
    return term_new_imp

# 用term和title，计算初始权重
init_weight_chaju = 2  # 参考值e，最好在 1~ 3之间，值越小，初始权重会根据点击title的分布，拉大分值差距。
query_termWeight = imp_init_weight(query_title_click, query_segs, stop_words, init_weight_chaju)

with open(data_path + "query_termWeight", 'w') as f:
    for query, w in query_termWeight.items():
        segs = query_segs[query]
        w = [round(a / sum(w), 3) for a in w]
        f.write(query + "\t" + str(segs) + "\t" + str(w) + "\n")

# 计算imp
loop = 3 # 迭代次数越多，计算越复杂，权重差距越大
max_term_num = 8 # 一个query最多8个term，太多会导致term权重偏低
imp_weight = imp(query_segs, query_title_click, query_termWeight, loop, max_term_num)

# # # 排序并输出
term_weight = sorted(imp_weight.items(), key=lambda item: item[1][0], reverse=True)

with open("../data/qa_data/term_weight_by_imp_l" + str(loop) + "_i" + str(init_weight_chaju) + "_d" + date, 'w') as f:
    for line in term_weight:
        line = line[0] + "\t" + str(line[1][0]) + "\t" + str(line[1][1]) + "\n"
        f.write(line)
