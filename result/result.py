import re
import numpy as np

def parse_metrics_from_line(line):
    """
    从每一行中提取 F1 Micro, F1 Macro, AUC 的数值
    """
    try:
        f1_micro = float(re.search(r'F1 Micro:\s*([\d.]+)', line).group(1))
        f1_macro = float(re.search(r'F1 Macro:\s*([\d.]+)', line).group(1))
        auc = float(re.search(r'AUC:\s*([\d.]+)', line).group(1))
        return f1_micro, f1_macro, auc
    except AttributeError:
        return None  # 这一行没有匹配成功，可能不是结果行

def analyze_file(filename):
    f1_micro_list = []
    f1_macro_list = []
    auc_list = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_metrics_from_line(line)
            if result:
                f1_micro, f1_macro, auc = result
                f1_micro_list.append(f1_micro)
                f1_macro_list.append(f1_macro)
                auc_list.append(auc)

    def summary(arr):
        arr = np.array(arr)
        return arr.mean(), arr.var()

    f1_micro_mean, f1_micro_var = summary(f1_micro_list)
    f1_macro_mean, f1_macro_var = summary(f1_macro_list)
    auc_mean, auc_var = summary(auc_list)

    print("==== 统计结果 ====")
    print(filename)
    print(f"F1 Micro: Mean = {f1_micro_mean:.4f}, Variance = {f1_micro_var:.4f}")
    print(f"F1 Macro: Mean = {f1_macro_mean:.4f}, Variance = {f1_macro_var:.4f}")
    print(f"AUC:       Mean = {auc_mean:.4f}, Variance = {auc_var:.4f}")




if __name__ == "__main__":
    # 替换为你的 txt 文件路径
    # file_map = ['./result/acm/train_gcn_acm.txt', './result/acm/train_gat_acm.txt', 
    #             './result/acm/train_m2v_acm.txt', './result/acm/train_hgt_acm.txt', 
    #             './result/acm/train_han_acm.txt', './result/freebase/train_gcn_freebase.txt', 
    #             './result/freebase/train_gat_freebase.txt', './result/freebase/train_m2v_freebase.txt', 
    #             './result/freebase/train_hgt_freebase.txt', './result/freebase/train_han_freebase.txt', 
    #             './result/dblp/train_gcn_dblp.txt', './result/dblp/train_gat_dblp.txt', 
    #             './result/dblp/train_m2v_dblp.txt', './result/dblp/train_hgt_dblp.txt', 
    #             './result/dblp/train_han_dblp.txt', './result/yelp/train_gcn_yelp.txt', 
    #             './result/yelp/train_gat_yelp.txt', './result/yelp/train_m2v_yelp.txt', 
    #             './result/yelp/train_hgt_yelp.txt', './result/yelp/train_han_yelp.txt']

    file_map = ['./result/refine/hgt_acm_refined.txt', './result/refine/hgt_dblp_refined.txt',
                './result/refine/hgt_freebase_refined.txt', './result/refine/hgt_yelp_refined.txt',
                './result/refine/han_acm_refined.txt', './result/refine/han_dblp_refined.txt']
    
    # file_map = ['./result/refine/han_acm_refined.txt', './result/refine/han_dblp_refined.txt']

    for i in range(6):
        analyze_file(file_map[i])
