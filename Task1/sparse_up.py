'''2211819 PP 04/25'''
import numpy as np
import psutil
import time
import os

# 读取文件，构建图
def read_file(file_path):
    graph={}
    nodes=[]
    with open(file_path,'r') as file:
        for line in file:
            from_node, to_node = map(int, line.strip().split())
            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append(to_node)
            if from_node not in nodes:
                nodes.append(from_node)
            if to_node not in nodes:
                nodes.append(to_node)
    return graph, nodes

# # 稀疏矩阵构建
# def build_sparse_matrix(graph, node_to_num, n):
#     sparse_matrix = {}
#     for from_node, to_nodes in graph.items():
#         from_idx = node_to_num[from_node]
#         out_degree = len(to_nodes)
#         if out_degree == 0:
#             continue
#         prob = 1.0 / out_degree
#         for to_node in to_nodes:
#             to_idx = node_to_num[to_node]
#             sparse_matrix[(to_idx, from_idx)] = prob
#     return sparse_matrix

# # 减少无效乘的sparse_mul
# def sparse_matvec_mul(sparse_matrix, vector, n):
#     result = [0.0] * n
#     for (i, j), value in sparse_matrix.items():
#         result[i] += value * vector[j]
#     return result

def pagerank(graph, nodes, alpha=0.85, tol=1e-10, max_iter=100):
    start_time = time.time()
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}
    index_to_node = {i: node for node, i in node_to_index.items()}

    # 初始化 PageRank 值
    pr = np.full(n, 1.0 / n)

    # 构建稀疏转移矩阵
    out_degree = np.zeros(n)
    adjacency = [[] for _ in range(n)]
    for from_node, to_nodes in graph.items():
        from_idx = node_to_index[from_node]
        out_degree[from_idx] = len(to_nodes)
        for to_node in to_nodes:
            to_idx = node_to_index[to_node]
            adjacency[from_idx].append(to_idx)

    for iteration in range(max_iter):
        pr_new = np.zeros(n)
        dangling_sum = pr[out_degree == 0].sum()
        for i in range(n):
            for j in adjacency[i]:
                pr_new[j] += pr[i] / out_degree[i]
        pr_new = alpha * pr_new + alpha * dangling_sum / n + (1 - alpha) / n
        if np.max(np.abs(pr_new - pr)) < tol:
            break
        pr = pr_new

    # 排序并提取前100个节点
    sorted_indices = np.argsort(-pr)
    sorted_results = [(index_to_node[i], pr[i]) for i in sorted_indices[:100]]

    end_time = time.time()
    print('时间开销：%.4f s' % (end_time - start_time))
    print('内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    return sorted_results


# 将结果输出到文件
def output_result(results, file_path):
    with open(file_path, 'w') as f:
        for node, score in results:
            f.write(f"{node} {score}\n")


if __name__ == '__main__':
    data_file_path = 'Data.txt'
    output_file_path = 'output/sparse_pr.txt'

    graph, nodes = read_file(data_file_path)
    pagerank_values = pagerank(graph, nodes)
    output_result(pagerank_values, output_file_path)