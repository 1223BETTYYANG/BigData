'''2211819 PP 04/25'''
import numpy as np
import psutil
import time
import os
from multiprocessing import Pool, cpu_count

# 读取文件，构建图
def read_file(file_path):
    graph = {}
    nodes = []
    with open(file_path, 'r') as file:
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

# 分块函数
def partition_nodes(nodes, num_blocks):
    nodes_sorted = sorted(nodes)
    block_size = (len(nodes_sorted) + num_blocks - 1) // num_blocks
    blocks = [nodes_sorted[i * block_size: (i + 1) * block_size] for i in range(num_blocks)]
    return blocks

# 单个块处理函数
def update_block(args):
    block, node_to_index, adjacency, out_degree, pr, n = args
    local_update = np.zeros(n)
    for node in block:
        i = node_to_index[node]
        if out_degree[i] == 0:
            continue
        for j in adjacency[i]:
            local_update[j] += pr[i] / out_degree[i]
    return local_update

# PageRank实现（分块 + 多进程版）
def pagerank(graph, nodes, alpha=0.85, tol=1e-10, max_iter=100, num_blocks=None):
    start_time = time.time()
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}
    index_to_node = {i: node for node, i in node_to_index.items()}

    # 初始化PageRank
    pr = np.full(n, 1.0 / n)

    # 构建邻接表与出度
    out_degree = np.zeros(n)
    adjacency = [[] for _ in range(n)]
    for from_node, to_nodes in graph.items():
        from_idx = node_to_index[from_node]
        out_degree[from_idx] = len(to_nodes)
        for to_node in to_nodes:
            to_idx = node_to_index[to_node]
            adjacency[from_idx].append(to_idx)

    # 确定分块数量
    if num_blocks is None:
        num_blocks = cpu_count()  # 默认用全部核心数
    blocks = partition_nodes(nodes, num_blocks)
    print(f"Total {num_blocks} blocks partitioned.")

    pool = Pool(processes=num_blocks)

    for iteration in range(max_iter):
        pr_new = np.zeros(n)
        dangling_sum = pr[out_degree == 0].sum()

        # 打包每个块需要传递的参数
        args_list = []
        for block in blocks:
            args_list.append((block, node_to_index, adjacency, out_degree, pr, n))

        # 并行处理各块
        results = pool.map(update_block, args_list)

        # 合并所有块的结果
        for local_update in results:
            pr_new += local_update

        # 加入随机跳转和悬挂节点贡献
        pr_new = alpha * pr_new + alpha * dangling_sum / n + (1 - alpha) / n

        # 收敛检测
        if np.max(np.abs(pr_new - pr)) < tol:
            print(f"Converged at iteration {iteration+1}")
            break

        pr = pr_new
        # print(f"Iteration {iteration+1} finished.")

    pool.close()
    pool.join()

    # 排序并提取前100个节点
    sorted_indices = np.argsort(-pr)
    sorted_results = [(index_to_node[i], pr[i]) for i in sorted_indices[:100]]

    end_time = time.time()
    print('总时间开销：%.4f s' % (end_time - start_time))
    print('内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    return sorted_results

# 将结果输出到文件
def output_result(results, file_path):
    with open(file_path, 'w') as f:
        for node, score in results:
            f.write(f"{node} {score}\n")

if __name__ == '__main__':
    data_file_path = 'Data.txt'
    output_file_path = 'output/block_multiprocess.txt'

    graph, nodes = read_file(data_file_path)
    pagerank_values = pagerank(graph, nodes, num_blocks=cpu_count())  
    output_result(pagerank_values, output_file_path)
