import numpy as np
import time
import psutil
import os
from collections import defaultdict
import math
import random

class SVDPP:
    def __init__(self, n_factors=50, learning_rate=0.01, reg_lambda=0.02, n_epochs=100):
        """
        SVD++算法实现
        
        参数:
        n_factors: 潜在因子数量
        learning_rate: 学习率
        reg_lambda: 正则化参数
        n_epochs: 训练轮数
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        
        # 模型参数
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.implicit_factors = None  # SVD++特有的隐式反馈因子
        
        # 数据存储
        self.user_items = defaultdict(set)  # 用户交互过的物品集合（隐式反馈）
        self.user_item_ratings = {}  # 用户-物品评分字典
        self.user_idx = {}  # 用户ID到索引的映射
        self.item_idx = {}  # 物品ID到索引的映射
        self.users = []
        self.items = []
        self.n_users = 0
        self.n_items = 0
        
    def load_train_data(self, train_file):
        """
        加载训练数据，与BiasSVD保持一致的数据结构
        """
        print("正在加载训练数据...")
        user_item_ratings = {}
        users = set()
        items = set()
        
        with open(train_file, 'r', encoding='utf-8') as f:
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
                
                # 构建隐式反馈
                self.user_items[user_id].add(item_id)
        
        self.user_item_ratings = user_item_ratings
        self.users = list(users)
        self.items = list(items)
        
        # 创建索引映射
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        self.item_idx = {i: j for j, i in enumerate(self.items)}
        
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        
        print(f"数据加载完成: {sum(len(items) for items in user_item_ratings.values())}条评分, {len(users)}个用户, {len(items)}个物品")
        
    def initialize_parameters(self):
        """
        初始化模型参数
        """
        # 计算全局平均评分
        all_ratings = [rating for user_ratings in self.user_item_ratings.values() 
                      for rating in user_ratings.values()]
        self.global_mean = np.mean(all_ratings)
        
        # 初始化偏置项
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        # 初始化潜在因子矩阵
        self.user_factors = np.random.normal(scale=0.01, size=(self.n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.01, size=(self.n_items, self.n_factors))
        
        # SVD++特有：隐式反馈因子矩阵
        self.implicit_factors = np.random.normal(scale=0.01, size=(self.n_items, self.n_factors))
        
    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分，与BiasSVD风格保持一致
        """
        # 如果用户或物品不在训练集中，返回全局均值
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.global_mean
            
        u_idx = self.user_idx[user_id]
        i_idx = self.item_idx[item_id]
        
        # 基础预测：全局均值 + 用户偏置 + 物品偏置
        prediction = self.global_mean + self.user_bias[u_idx] + self.item_bias[i_idx]
        
        # 用户潜在因子
        user_factor = self.user_factors[u_idx].copy()
        
        # SVD++核心：加入隐式反馈
        user_implicit_items = list(self.user_items[user_id])
        if user_implicit_items:
            # 计算隐式反馈的影响
            implicit_sum = np.zeros(self.n_factors)
            for implicit_item in user_implicit_items:
                if implicit_item in self.item_idx:
                    implicit_sum += self.implicit_factors[self.item_idx[implicit_item]]
            
            # 归一化
            implicit_feedback = implicit_sum / math.sqrt(len(user_implicit_items))
            user_factor += implicit_feedback
        
        # 最终预测
        prediction += np.dot(user_factor, self.item_factors[i_idx])
        
        # 限制评分范围
        return np.clip(prediction, 0.0, 100.0)
    
    def train(self):
        """
        训练SVD++模型
        """
        print("开始训练SVD++模型...")
        start_time = time.time()
        
        self.initialize_parameters()
        
        # 构建训练数据列表
        training_data = []
        for user_id, items in self.user_item_ratings.items():
            for item_id, rating in items.items():
                training_data.append((user_id, item_id, rating))
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            
            # 随机打乱训练数据
            random.shuffle(training_data)
            
            for user_id, item_id, rating in training_data:
                u_idx = self.user_idx[user_id]
                i_idx = self.item_idx[item_id]
                
                # 预测评分
                pred = self.predict(user_id, item_id)
                error = rating - pred
                epoch_loss += error ** 2
                
                # 计算梯度并更新参数
                # 保存当前参数用于更新
                user_bias_old = self.user_bias[u_idx]
                item_bias_old = self.item_bias[i_idx]
                user_factor_old = self.user_factors[u_idx].copy()
                item_factor_old = self.item_factors[i_idx].copy()
                
                # 计算隐式反馈
                user_implicit_items = list(self.user_items[user_id])
                implicit_feedback = np.zeros(self.n_factors)
                if user_implicit_items:
                    for implicit_item in user_implicit_items:
                        if implicit_item in self.item_idx:
                            implicit_feedback += self.implicit_factors[self.item_idx[implicit_item]]
                    implicit_feedback /= math.sqrt(len(user_implicit_items))
                
                # 更新偏置项
                self.user_bias[u_idx] += self.learning_rate * (error - self.reg_lambda * user_bias_old)
                self.item_bias[i_idx] += self.learning_rate * (error - self.reg_lambda * item_bias_old)
                
                # 更新潜在因子
                self.user_factors[u_idx] += self.learning_rate * (error * item_factor_old - self.reg_lambda * user_factor_old)
                self.item_factors[i_idx] += self.learning_rate * (error * (user_factor_old + implicit_feedback) - self.reg_lambda * item_factor_old)
                
                # 更新隐式反馈因子
                if user_implicit_items:
                    implicit_gradient = error * item_factor_old / math.sqrt(len(user_implicit_items))
                    for implicit_item in user_implicit_items:
                        if implicit_item in self.item_idx:
                            self.implicit_factors[self.item_idx[implicit_item]] += self.learning_rate * (implicit_gradient - self.reg_lambda * self.implicit_factors[self.item_idx[implicit_item]])
            
            # 计算RMSE
            rmse = math.sqrt(epoch_loss / len(training_data))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f}秒")
        return training_time

# 与您的BiasSVD风格保持一致的辅助函数
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

def split_train_test(user_item_ratings, test_ratio=1.0):
    """
    随机将每个用户的评分记录划分为训练集和测试集
    与BiasSVD保持一致
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
    
    return train_data, test_data

def calculate_rmse(test_data, model):
    """
    计算RMSE - 改进版本，与您的原始函数风格保持一致
    """
    total_error = 0
    count = 0
    
    for user in test_data:
        for item, true_rating in test_data[user].items():
            pred = model.predict(user, item)
            total_error += (pred - true_rating) ** 2
            count += 1
    
    return np.sqrt(total_error / count) if count > 0 else 0

def calculate_rmse_on_training(model):
    """
    计算训练集上的RMSE
    """
    total_error = 0
    count = 0
    
    for user in model.user_item_ratings:
        for item in model.user_item_ratings[user]:
            true_rating = model.user_item_ratings[user][item]
            predicted_rating = model.predict(user, item)
            total_error += (true_rating - predicted_rating) ** 2
            count += 1
    
    return np.sqrt(total_error / count) if count > 0 else 0

def save_results_svd_plus_plus(filename, test_data, model):
    """
    保存SVD++预测结果，与BiasSVD风格保持一致
    """
    start_time = time.time()  # 记录预测开始时间
    mem_before_predict = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 预测前内存

    with open(filename, 'w', encoding='utf-8') as f:
        for user in test_data:
            items = test_data[user]
            f.write(f"{user}|{len(items)}\n")
            for item in items:
                score = model.predict(user, item)
                f.write(f"{item}\t{score:.0f}\n")

    end_time = time.time()  # 记录预测结束时间
    mem_after_predict = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 预测后内存

    predict_time = end_time - start_time
    predict_memory = mem_after_predict - mem_before_predict

    print(f"预测时间: {predict_time:.4f} 秒")
    print(f"预测内存消耗: {predict_memory:.4f} MB")
    print(f"预测结果已保存到: {filename}")
    
    return predict_time, predict_memory

def get_memory_usage():
    """
    获取当前内存使用量（MB）
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024

def main():
    """
    主函数：完整的SVD++实验流程
    """
    print("=" * 60)
    print("SVD++推荐算法实验")
    print("=" * 60)
    
    # 数据文件路径
    train_file = "data/Train.txt"
    test_file = "data/Test.txt"
    output_file = "data/SVD++_predictions.txt"
    
    # 记录初始内存使用
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 创建SVD++模型
    model = SVDPP(
        n_factors=50,
        learning_rate=0.005,
        reg_lambda=0.02,
        n_epochs=100
    )
    
    # 加载训练数据
    model.load_train_data(train_file)
    
    # 记录训练前内存使用
    before_training_memory = get_memory_usage()
    print(f"加载数据后内存使用: {before_training_memory:.2f} MB")
    
    # 训练模型
    training_time = model.train()
    
    # 记录训练后内存使用
    after_training_memory = get_memory_usage()
    print(f"训练后内存使用: {after_training_memory:.2f} MB")
    
    # 计算训练集RMSE
    train_rmse = calculate_rmse_on_training(model)
    print(f"训练集RMSE: {train_rmse:.4f}")
    
    # 如果需要在部分训练数据上测试（类似您的原始函数）
    # 可以使用split_train_test函数
    train_data, internal_test_data = split_train_test(model.user_item_ratings)
    
    # 在内部测试集上计算RMSE
    internal_test_rmse = calculate_rmse(internal_test_data, model)
    print(f"内部测试集RMSE: {internal_test_rmse:.4f}")
    
    # 加载外部测试数据并生成预测
    test_data = load_test_data(test_file)
    predict_time, predict_memory = save_results_svd_plus_plus(output_file, test_data, model)
    
    # 计算最终内存使用
    final_memory = get_memory_usage()
    memory_consumption = final_memory - initial_memory
    
    # 输出实验结果
    print("\n" + "=" * 60)
    print("实验结果统计")
    print("=" * 60)
    print(f"算法: SVD++")
    print(f"潜在因子数: {model.n_factors}")
    print(f"学习率: {model.learning_rate}")
    print(f"正则化参数: {model.reg_lambda}")
    print(f"训练轮数: {model.n_epochs}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"预测时间: {predict_time:.4f} 秒")
    print(f"总内存消耗: {memory_consumption:.2f} MB")
    print(f"预测内存消耗: {predict_memory:.4f} MB")
    print(f"用户数量: {model.n_users}")
    print(f"物品数量: {model.n_items}")
    print(f"评分数量: {sum(len(items) for items in model.user_item_ratings.values())}")
    print(f"数据稀疏度: {sum(len(items) for items in model.user_item_ratings.values()) / (model.n_users * model.n_items) * 100:.4f}%")
    print(f"训练集RMSE: {train_rmse:.4f}")
    print(f"内部测试集RMSE: {internal_test_rmse:.4f}")
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()