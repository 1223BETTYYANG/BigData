import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import random
import time
import psutil
import os
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F


class LightGCNModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCNModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices, edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        user_embs = [user_emb]
        item_embs = [item_emb]

        for _ in range(self.num_layers):
            user_agg = torch.zeros_like(user_emb)
            item_agg = torch.zeros_like(item_emb)
            u_nodes, i_nodes = edge_index
            user_agg.index_add_(0, u_nodes, item_emb[i_nodes])
            item_agg.index_add_(0, i_nodes, user_emb[u_nodes])
            user_emb = F.normalize(user_agg, p=2, dim=1)
            item_emb = F.normalize(item_agg, p=2, dim=1)
            user_embs.append(user_emb)
            item_embs.append(item_emb)

        final_user_emb = torch.mean(torch.stack(user_embs), dim=0)
        final_item_emb = torch.mean(torch.stack(item_embs), dim=0)

        return torch.sum(final_user_emb[user_indices] * final_item_emb[item_indices], dim=1)


def load_train_data(file):
    user_item_dict = defaultdict(set)
    user_item_ratings = {}
    rating_matrix = []
    users, items = set(), set()

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
            user_item_dict[user_id].add(item_id)
            user_item_ratings[user_id][item_id] = score
            rating_matrix.append((user_id, item_id, score))
            users.add(user_id)
            items.add(item_id)

    return user_item_dict, user_item_ratings, rating_matrix, list(users), list(items)


def split_train_test(user_item_ratings):
    train_data, test_data = {}, {}
    for user, items in user_item_ratings.items():
        all_items = list(items.items())
        test_item = random.choice(all_items)
        test_data[user] = {test_item[0]: test_item[1]}
        train_data[user] = {item: rating for item, rating in all_items if item != test_item[0]}
    return train_data, test_data


def load_test_data(file):
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
        test_data[user_id] = [lines[idx + i].strip() for i in range(n_items)]
        idx += n_items
    return test_data


def prepare_training_data(user_item_ratings, user_idx, item_idx):
    users, items, ratings, edges = [], [], [], []
    for u, item_dict in user_item_ratings.items():
        for i, r in item_dict.items():
            if u in user_idx and i in item_idx:
                uid, iid = user_idx[u], item_idx[i]
                users.append(uid)
                items.append(iid)
                ratings.append(r)
                edges.append([uid, iid])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    dataset = TensorDataset(torch.tensor(users), torch.tensor(items), torch.tensor(ratings, dtype=torch.float32))
    return dataset, edge_index


def train_lightgcn(user_item_dict, user_item_ratings, users, items,
                   embedding_dim=64, num_layers=3, steps=200, lr=0.001,
                   weight_decay=1e-5, batch_size=1024, device='cpu'):
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {i: j for j, i in enumerate(items)}
    num_users, num_items = len(users), len(items)

    dataset, edge_index = prepare_training_data(user_item_ratings, user_idx, item_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LightGCNModel(num_users, num_items, embedding_dim, num_layers).to(device)
    edge_index = edge_index.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    print(f"开始训练 LightGCN 模型（设备: {device}）")
    max_memory_usage = 0
    for epoch in range(steps):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch_users, batch_items, batch_ratings in train_loader:
            batch_users, batch_items, batch_ratings = batch_users.to(device), batch_items.to(device), batch_ratings.to(device)
            optimizer.zero_grad()
            predictions = model(batch_users, batch_items, edge_index)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        if epoch % 10 == 0 or epoch == steps - 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss / len(train_loader):.6f} | Time: {epoch_time:.2f}s")
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        max_memory_usage = max(max_memory_usage, current_memory)
    print(f"训练过程最大内存峰值: {max_memory_usage:.2f} MB")
    return model, user_idx, item_idx


def predict_lightgcn(model, user_idx, item_idx, user, item, edge_index, device='cpu'):
    if user not in user_idx or item not in item_idx:
        return 50.0
    model.eval()
    with torch.no_grad():
        uid = torch.tensor([user_idx[user]], device=device)
        iid = torch.tensor([item_idx[item]], device=device)
        score = model(uid, iid, edge_index.to(device)).item()
    return np.clip(score, 0, 100)


def save_results_lightgcn(filename, test_data, model, user_idx, item_idx, edge_index, device='cpu'):
    # 记录预测开始时间和内存
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    with open(filename, 'w', encoding='utf-8') as f:
        for user in test_data:
            items = test_data[user]
            f.write(f"{user}|{len(items)}\n")
            for item in items:
                score = predict_lightgcn(model, user_idx, item_idx, user, item, edge_index, device)
                f.write(f"{item} {int(score)}\n")
    # 记录预测结束时间和内存
    end_time = time.time()
    mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    print(f"预测耗时: {end_time - start_time:.2f} 秒")
    print(f"预测内存占用: {mem_after - mem_before:.2f} MB")

def save_model_lightgcn(model, user_idx, item_idx, edge_index, model_file='lightgcn_model.pkl'):
    data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_users': model.num_users,
            'num_items': model.num_items,
            'embedding_dim': model.embedding_dim,
            'num_layers': model.num_layers
        },
        'user_idx': user_idx,
        'item_idx': item_idx,
        'edge_index': edge_index.cpu()
    }
    with open(model_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"模型已保存: {model_file}")


def load_model_lightgcn(model_file='lightgcn_model.pkl', device='cpu'):
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    cfg = data['model_config']
    model = LightGCNModel(cfg['num_users'], cfg['num_items'], cfg['embedding_dim'], cfg['num_layers'])
    model.load_state_dict(data['model_state_dict'])
    model.to(device)
    return model, data['user_idx'], data['item_idx'], data['edge_index'].to(device)


def calculate_rmse(user_item_ratings, model, user_idx, item_idx, edge_index, device='cpu'):
    model.eval()
    total_error = 0
    count = 0
    with torch.no_grad():
        for user in user_item_ratings:
            for item in user_item_ratings[user]:
                true = user_item_ratings[user][item]
                pred = predict_lightgcn(model, user_idx, item_idx, user, item, edge_index, device)
                total_error += (true - pred) ** 2
                count += 1
    return np.sqrt(total_error / count)


if __name__ == '__main__':
    train_file = './data/train.txt'
    test_file = './data/test.txt'
    output_file = './output/LightGCN.txt'
    model_file = './model/lightgcn_model.pkl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")
    if torch.cuda.is_available():
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

    user_item_dict, user_item_ratings, rating_matrix, users, items = load_train_data(train_file)
    test_data = load_test_data(test_file)
    train_data, test_data_from_train = split_train_test(user_item_ratings)

    try:
        model, user_idx, item_idx, edge_index = load_model_lightgcn(model_file, device)
        print("模型已加载")
    except FileNotFoundError:
        print("模型未找到，开始训练")
        train_start_time = time.time()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        model, user_idx, item_idx = train_lightgcn(
            user_item_dict, train_data, users, items,
            embedding_dim=64, num_layers=3, steps=100,
            lr=0.001, batch_size=1024, device=device
        )
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"训练总耗时: {train_time:.2f}s")
        print(f"训练总内存占用: {mem_after - mem_before:.2f} MB")
        _, edge_index = prepare_training_data(train_data, user_idx, item_idx)
        edge_index = edge_index.to(device)
        save_model_lightgcn(model, user_idx, item_idx, edge_index, model_file)
        

    rmse = calculate_rmse(test_data_from_train, model, user_idx, item_idx, edge_index, device)
    print(f"测试集 RMSE: {rmse:.4f}")

    save_results_lightgcn(output_file, test_data, model, user_idx, item_idx, edge_index, device)
    print(f"预测结果已保存至 {output_file}")
