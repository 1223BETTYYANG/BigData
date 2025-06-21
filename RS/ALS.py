import numpy as np
import pickle
import random
import time
import psutil
import os


def load_train_data(file):
    """
    读取训练文件，返回user-item评分字典，用户和物品ID列表
    """
    user_item_ratings = {}
    users = set()
    items = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        user_id, n_items = line.split('|')
        n_items = int(n_items)
        idx += 1
        user_item_ratings[user_id] = {}
        for _ in range(n_items):
            item_line = lines[idx].strip()
            idx += 1
            item_id, score = item_line.split()
            score = float(score)
            user_item_ratings[user_id][item_id] = score
            users.add(user_id)
            items.add(item_id)
    return user_item_ratings, list(users), list(items)

def split_train_test(user_item_ratings, test_ratio=1.0):
    """
    随机将每个用户的评分记录划分为训练集和测试集
    其中，test_ratio表示每个用户随机选择多少比例的评分记录作为测试集（默认选择1条作为测试集）
    返回训练集和测试集
    """
    train_data = {}
    test_data = {}
    
    for user, items in user_item_ratings.items():
        all_items = list(items.items())  # (item_id, rating) pair
        # 随机选择一条作为测试集
        test_item = random.choice(all_items)
        test_data[user] = {test_item[0]: test_item[1]}  # {item_id: rating}
        
        # 剩下的作为训练集
        train_data[user] = {item: rating for item, rating in all_items if item != test_item[0]}
    
    return train_data, test_data  # 返回训练集和测试集


def load_test_data(file):
    """
    读取测试文件，返回用户和预测的物品列表
    """
    test_data = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        user_id, n_items = line.split('|')
        n_items = int(n_items)
        idx += 1
        test_data[user_id] = []
        for _ in range(n_items):
            item_id = lines[idx].strip()
            idx += 1
            test_data[user_id].append(item_id)
    return test_data

def train_als(user_item_ratings, users, items, K=50, steps=200, lambda_reg=0.1):
    """
    基于 ALS (Alternating Least Squares) 的矩阵分解训练函数
    """
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {i: j for j, i in enumerate(items)}

    n_users = len(users)
    n_items = len(items)

    # 初始化潜在因子矩阵
    P = np.random.normal(scale=0.1, size=(n_users, K)).astype(np.float64)
    Q = np.random.normal(scale=0.1, size=(n_items, K)).astype(np.float64)

    # 构建评分矩阵
    R = np.zeros((n_users, n_items))
    for u in user_item_ratings:
        for i in user_item_ratings[u]:
            R[user_idx[u]][item_idx[i]] = user_item_ratings[u][i]

    # 构建掩码矩阵（标记哪些位置有评分）
    mask = R > 0

    total_train_time = 0  # 总训练时间
    total_round_time = 0  # 每轮训练平均时间
    num_rounds = 0  # 训练轮数
    max_memory_usage = 0  # 最大内存追踪变量

    for step in range(steps):
        round_start_time = time.time()
        round_memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        num_rounds += 1

        # 固定Q，更新P
        for u in range(n_users):
            idx = mask[u]  # 用户u评分过的物品
            if np.sum(idx) == 0:
                continue
            Q_i = Q[idx]  # 用户u评分过的物品的潜在因子
            R_u = R[u, idx]  # 用户u的评分向量
            A = Q_i.T @ Q_i + lambda_reg * np.eye(K)
            V = Q_i.T @ R_u
            P[u] = np.linalg.solve(A, V)

        # 固定P，更新Q
        for i in range(n_items):
            idx = mask[:, i]  # 对物品i评分过的用户
            if np.sum(idx) == 0:
                continue
            P_u = P[idx]  # 对物品i评分过的用户的潜在因子
            R_i = R[idx, i]  # 物品i的评分向量
            A = P_u.T @ P_u + lambda_reg * np.eye(K)
            V = P_u.T @ R_i
            Q[i] = np.linalg.solve(A, V)

        round_end_time = time.time()
        round_memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        round_time = round_end_time - round_start_time
        round_memory = round_memory_after - round_memory_before

        # 累加单轮训练时间
        total_train_time += round_time
        total_round_time += round_time 
        max_memory_usage = max(max_memory_usage, round_memory)  # 更新最大值

        if step % 5 == 0 or step == steps - 1:
            # 计算损失函数
            loss = 0
            for u in range(n_users):
                for i in range(n_items):
                    if mask[u, i]:
                        pred = np.dot(P[u], Q[i])
                        loss += (R[u, i] - pred) ** 2
            # 添加正则化项
            loss += lambda_reg * (np.sum(P ** 2) + np.sum(Q ** 2))
            print(f"Step {step}, loss={loss:.6f}")

    avg_round_time = total_round_time / num_rounds
    print(f"训练总耗时: {total_train_time:.4f}秒")
    print(f"单轮训练平均耗时: {avg_round_time:.4f}秒")
    print(f"训练过程最大内存峰值: {max_memory_usage:.4f} MB")

    return P, Q, user_idx, item_idx

def predict_als(P, Q, user_idx, item_idx, user, item, min_score=0.0, max_score=100.0):
    """
    预测用户对物品的评分，并限制评分范围在 min_score 和 max_score 之间。
    如果用户或物品未出现，则返回基于全局均值的预测值。
    """
    if user not in user_idx or item not in item_idx:
        # 如果用户或物品未出现，使用平均潜在因子预测
        return np.clip(np.dot(np.mean(P, axis=0), np.mean(Q, axis=0)), min_score, max_score)
    
    u_idx = user_idx[user]
    i_idx = item_idx[item]
    score = np.dot(P[u_idx], Q[i_idx])
    # 限制评分范围
    return np.clip(score, min_score, max_score)

def save_results_als(filename, test_data, P, Q, user_idx, item_idx):
    start_time = time.time()  # 记录预测开始时间
    mem_before_predict = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 预测前内存

    with open(filename, 'w', encoding='utf-8') as f:
        for user in test_data:
            items = test_data[user]
            f.write(f"{user}|{len(items)}\n")
            for item in items:
                score = predict_als(P, Q, user_idx, item_idx, user, item)
                f.write(f"{item} {score:.6f}\n")

    end_time = time.time()  # 记录预测结束时间
    mem_after_predict = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 预测后内存

    predict_time = end_time - start_time
    predict_memory = mem_after_predict - mem_before_predict

    print(f"预测耗时: {predict_time:.4f}秒")
    print(f"预测内存占用: {predict_memory:.4f} MB")

def save_model_als(P, Q, user_idx, item_idx, model_file='als_model.pkl'):
    """
    保存训练的模型到文件。
    """
    with open(model_file, 'wb') as f:
        pickle.dump({
            'P': P,
            'Q': Q,
            'user_idx': user_idx,
            'item_idx': item_idx
        }, f)
    print(f"ALS 模型已保存到 {model_file}")

def load_model_als(model_file='als_model.pkl'):
    """
    从文件加载训练好的 ALS 模型。
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model['P'], model['Q'], model['user_idx'], model['item_idx']

def calculate_rmse(user_item_ratings, P, Q, user_idx, item_idx):
    """
    计算基于训练数据的均方根误差（RMSE）
    """
    error = 0
    count = 0
    for user in user_item_ratings:
        for item in user_item_ratings[user]:
            true_rating = user_item_ratings[user][item]
            predicted_rating = predict_als(P, Q, user_idx, item_idx, user, item)
            error += (true_rating - predicted_rating) ** 2
            count += 1
    rmse = np.sqrt(error / count)
    return rmse


if __name__ == '__main__':
    train_file = './data/train.txt'  # 训练数据文件路径
    test_file = './data/test.txt'    # 测试数据文件路径
    output_file = 'output/ALS.txt'  # 预测结果保存路径
    model_file = 'model/als_model.pkl'  # 模型文件路径

    # 加载训练数据 (train.txt)
    user_item_ratings, users, items = load_train_data(train_file)

    # 加载测试数据 (test.txt)
    test_data = load_test_data(test_file)

    # 将训练数据随机划分为训练集和测试集
    train_data, test_data_from_train = split_train_test(user_item_ratings)

    try:
        # 尝试加载模型
        P, Q, user_idx, item_idx = load_model_als(model_file)
        print(f"已加载模型: {model_file}")

        # 使用训练数据划分后的测试集计算RMSE
        rmse = calculate_rmse(test_data_from_train, P, Q, user_idx, item_idx)
        print(f"测试集的RMSE: {rmse:.6f}")
    
    except FileNotFoundError:
        print(f"模型文件 {model_file} 未找到，正在训练模型...")

        # 训练模型
        total_memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"训练前内存占用: {total_memory_before:.4f} MB")

        P, Q, user_idx, item_idx = train_als(train_data, users, items)

        total_memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"训练后内存占用: {total_memory_after:.4f} MB")
        print(f"训练总内存占用: {total_memory_after - total_memory_before:.4f} MB")

        # 保存训练好的模型
        save_model_als(P, Q, user_idx, item_idx, model_file)

        # 使用训练数据划分后的测试集计算RMSE
        rmse = calculate_rmse(test_data_from_train, P, Q, user_idx, item_idx)
        print(f"训练集划分后的测试集RMSE: {rmse:.6f}")

    # 使用test.txt数据进行预测并输出结果
    save_results_als(output_file, test_data, P, Q, user_idx, item_idx)
    print(f"预测完成，结果保存在 {output_file}")

    # 输出最终的RMSE
    print(f"最终测试集的RMSE: {rmse:.6f}")