import argparse
import itertools
import math
import networkx as nx
import numpy as np
import time
from collections import defaultdict
from disjoint_set import DisjointSet
from mbi import Dataset, Domain, CliqueVector
from mbi.inference import FactoredInference
from mbi.graphical_model import GraphicalModel
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.special import logsumexp
import pandas as pd
import multiprocessing
from cdp2adp import cdp_rho
import random
import networkx as nx
from heapq import heappush, heappop
from itertools import count
import copy
import dict2obj
import math
import itertools as it
from functools import reduce
import numpy as np


privacy = []    # 记录隐私预算分配
target = []     # 记录选择哪些边缘
Marginal_size = []  # 记录所有边缘的属性域大小
InDif = []      # 记录属性间相关性


"""
该版本提供分布的多进程学习+多进程一致性处理
该版本提供新的噪音度量选择方法
该版本提供属性间相关性推理方法
"""



def _weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


# 关键 dijkstra会中找出边缘最小的边
# 基于 dijkstra 改写算法，找出相关性最大的边缘
def _correlation_multisource(
    G, sources, weight, pred=None,  paths=None, target=None):
    G_succ = G._succ if G.is_directed() else G._adj
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    # 为所有本身的属性相关性赋值为1
    for source in sources:
        if source not in G:
            print('not found source', source)
            break
        seen[source] = 1
        push(fringe, (0, next(c), source))

    while fringe:
        (d, _, v) = pop(fringe)
        d = 1 - d   # 记录
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():  # u表示与v相连的节点 e表示权重
            cost = weight(v, u, e)  # 开销
            if cost is None:
                continue
            vu_dist = dist[v] * cost    # vu_dist 从出发点经过v到u的相关性
            if u in dist:
                u_dist = dist[u]    # u_dist表示从出发点到u的最终相关性
                if vu_dist > u_dist:
                    print('当前路径的相关性比计算出的相关性更大（出错）')
            elif u not in seen or vu_dist > seen[u]:
                seen[u] = vu_dist   # 到达阶段u时的相关性
                push(fringe, (1 - vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    dist.pop(source)
    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return (dist, paths)


def multi_source_correlation(G, sources, target=None, cutoff=None, weight="weight"):
    if not sources:
        raise ValueError("sources must not be empty")
    if target in sources:
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist, paths = _correlation_multisource(
        G, sources, weight, paths=paths, target=target
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths)
    except KeyError as e:
        raise nx.NetworkXNoPath(f"No path to {target}.") from e


def find_max_cor(dif, Org_cor):
    cor = 0
    for k in dif.keys():
        for u, e in dif[k].items():
            if e["weight"] > cor:
                node_1 = k
                node_2 = u
                cor = e["weight"]
    print('插入相关性边缘', (node_1, node_2, Org_cor[node_1][node_2]['weight']))
    return [(node_1, node_2, Org_cor[node_1][node_2]['weight'])]


def update_dif(cur, org):
    differ = copy.deepcopy(org)
    current = copy.deepcopy(cur)
    for k in current.keys():
        for u, e in current[k].items():
            differ[k][u]['weight'] = differ[k][u]['weight'] * (1 - current[k][u]['weight'])
            current[k][u]['weight'] -= current[k][u]['weight']
    return differ


def subProcess(q, tuple_cliques, iters, total):  # 多进程函数 每个进程各自的合成数据集没问题
    # 多进程函数
    sub_data = []  # 用于记录训练出来的团分布
    for tuple_clique in tuple_cliques:
        engine_fully = FactoredInference(tuple_clique[0].domain, iters=iters)
        engine_fully.estimate(tuple_clique[1], total=total)
        synth = engine_fully.model.synthetic_data()
        sub_data.append(synth)
    q.put(sub_data)


def multi_process_measure(log1, log2, engine, total, data):  # 多进程数据收集算法
    # 贪心算法为每个多进程尽量分配较为平均的属性域大小的团
    iters = 1000
    log2_cliques = []  # 记录每个团二维分布的列表
    data_cliques = []  # 记录每个团包含的属性
    range_count = []  # 记录每个团的属性域大小

    # 对于每个团，记录团包含的属性以及分布
    for clique in engine.model.cliques:
        data_clique = clique_data(data, clique)
        log2_clique = clique_log_v2(data_clique, log1, log2)
        log2_cliques.append(log2_clique)
        data_cliques.append(data_clique)
        range_count.append(engine.model.domain.project(clique).size())

    num_process = min([16, len(engine.model.cliques)])
    count_range_list = [0 for i in range(num_process)]  # 用于记录当前每个进程需要处理的属性域大小
    cliques_list = []  # 记录各个进程需要学习的团
    for i in range(num_process):
        cliques_list.append([])

    print('range', range_count)
    # 根据团的域值大小，通过贪心算法分配给每个进程，然后调用多进程学习每个团的分布
    for i in range(len(log2_cliques)):
        process_index = count_range_list.index(min(count_range_list))  # 记录对哪个进程添加当前团数据
        clique_index = range_count.index(max(range_count))  # 记录当前需要添加哪个团
        count_range_list[process_index] += max(range_count)
        range_count[range_count.index(max(range_count))] = -1
        cliques_list[process_index].append(
            (data_cliques[clique_index], log2_cliques[clique_index]))
    print('count_range_list', count_range_list)

    sub_data = []

    q = multiprocessing.Queue()  # q用于进程间通信 获取子进程计算得到的数据样本
    jobs = []  # 用于阻塞主进程
    # 开启所有子进程
    for i in range(num_process):
        th = multiprocessing.Process(target=subProcess,
                                     args=(q, cliques_list[i], iters, total))
        jobs.append(th)
        th.start()
    for i in jobs:
        while i.is_alive():
            while not q.empty():
                sub_data = sub_data + q.get()  # 取出所有子进程的采样结果
    for i in jobs:
        i.join()

    print("join finish")  # join()用于阻塞主进程的执行  等待子进程执行完毕后 继续住进程
    print("is q empty?", q.empty())

    sub_data_sort = []
    for clique in engine.model.cliques:
        for i in range(len(sub_data)):
            if sub_data[i].domain.attrs == clique:
                sub_data_sort.append(sub_data[i])

    return sub_data_sort



def MST(data, epsilon, delta):
    time_start = time.time()
    rho = cdp_rho(epsilon, delta)
    # sigma = np.sqrt(5.0 / (9.0 * rho))  # 每次添加噪音分得 9/10的隐私预算
    sigma1 = np.sqrt(5 / rho)  # 1/10的隐私预算用于度量原始数据集的一维分布
    cliques = [(col,) for col in data.domain]
    log1 = measure_single(data, cliques, sigma1)  # 收集原始数据集的一维属性信息用于属性压缩

    # 计算临时样本总数
    variances = np.array([])
    estimates = np.array([])
    for Q, y, noise, proj in log1:
        o = np.ones(Q.shape[1])
        v = lsmr(Q.T, o, atol=0, btol=0)[0]
        if np.allclose(Q.T.dot(v), o):
            variances = np.append(variances, noise ** 2 * np.dot(v, v))  # 方差*桶数量
            estimates = np.append(estimates, np.dot(v, y))  # 估计样本总数
    if estimates.size == 0:
        total_tmp = 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        total_tmp = max(1, estimate)
    total_tmp = int(total_tmp)


    data, log1, undo_compress_fn = compress_domain(data, log1, total_tmp)

    cliques = select_new(data, rho / 10.0, rho * 8.0 / 10.0)
    sigma_clique = []

    # 根据选择的边缘分布属性域 分配隐私预算
    sum_size = 0
    for j in cliques:
        sum_size += math.pow(data.domain.project(j).size(), 2/3)

    for i in cliques:
        sigma_clique.append(np.sqrt((5.0 * sum_size) / (rho * 8.0 * math.pow(data.domain.project(i).size(), 2/3))))

    print('sigma', sigma_clique)
    log2 = measure(data, cliques, sigma_clique)

    time_end = time.time()
    print('select and measure cost', time_end - time_start)

    time_start = time.time()

    total, log1, log2 = consistent_process(data, log1, log2)

    engine = FactoredInference(data.domain, iters=1000)
    engine._setup(log2, total=total)

    fully_cliques = check_if_fully_clique(engine.model.cliques)
    print('number of model cliques', len(fully_cliques))
    sub_data = multi_process_measure(log1, log2, engine, total, data)

    time_end = time.time()
    print('parallel cost', time_end - time_start)

    print('sub_data size', len(sub_data))

    time_start = time.time()
    synth_data = sub_data_combine(sub_data)
    time_end = time.time()
    print('combine cost', time_end - time_start)

    return undo_compress_fn(synth_data), cliques
    # return synth_data, cliques


def consistent_process(data, log1, log2):
    # 计算样本总数
    variances = np.array([])
    estimates = np.array([])
    for Q, y, noise, proj in (log1 + log2):
        o = np.ones(Q.shape[1])
        v = lsmr(Q.T, o, atol=0, btol=0)[0]
        if np.allclose(Q.T.dot(v), o):
            variances = np.append(variances, noise ** 2 * np.dot(v, v))  # 方差*桶数量
            estimates = np.append(estimates, np.dot(v, y))  # 估计样本总数
    if estimates.size == 0:
        total = 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        total = max(1, estimate)
    total = int(total)
    print('total', total)

    # 计算一致性处理前的一维分布误差
    errors = []
    for Q, y, noise, proj in log1:
        X = data.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - y / y.sum(), 1)
        errors.append(e)
    print('Average 1-way Error : ', np.mean(errors))

    # 度量边缘一致性处理
    index = 0

    # 首先保证边缘分布的总数等于total
    # 一维边缘的一致性
    for marg in log1:
        error = total - sum(marg[1])
        wgt_marg = marg[1] + error / np.shape(marg[1])
        log1[index] = (marg[0], wgt_marg, marg[2], marg[3])
        index += 1

    # 二路边缘的一致性处理
    index = 0
    for marg in log2:
        error = total - sum(marg[1])
        wgt_marg = marg[1] + error / np.shape(marg[1])
        log2[index] = (marg[0], wgt_marg, marg[2], marg[3])
        index += 1

    # groups记录包含 每个属性 的边缘 的集合
    measures = log2
    groups = defaultdict(lambda: [])
    for col in data.domain:
        index = 0
        for marg in measures:
            if col in marg[3] and len(marg[3]) > 1:
                colshape = data.project(marg[3]).domain.shape
                m = (marg[1], marg[2], marg[3], colshape, index)  # 记录噪音边缘的分布，噪音大小，边缘名称，每个属性的域, 在噪音分布集合中的位置
                groups[col].append(m)
            index += 1

    idx_add = defaultdict(lambda: [])
    col_add = defaultdict(lambda: [])
    for col in data.domain:
        index = 0
        for marg in measures:
            if col in marg[3] and len(marg[3]) == 1:
                idx_add[col].append(1)
                col_add[col].append(marg)
            index += 1

    col_index = 0

    # 对每个属性计算方差最小化的一致性分布
    for col in data.domain:
        lens = data.project(col).domain.shape[0]

        if idx_add[col] == [1]:
            w = np.zeros((2, lens))  # 创建0矩阵列表，用于统计投影后的一维分布
            variances = np.zeros(2)  # 记录一维分布中 每组数据对应的方差
            # 添加单个团的数据分布与方差
            w[0] = col_add[col][0][1]
            variances[0] = col_add[col][0][2] ** 2

        else:
            w = np.zeros((len(groups[col]) + 1, lens))  # 创建0矩阵列表，用于统计投影后的一维分布
            variances = np.zeros(len(groups[col]) + 1)  # 记录一维分布中 每组数据对应的方差

        # 统一一维分布与噪音
        for marg in log1:
            if col in marg[3]:
                w[len(w) - 1] = marg[1]
                variances[len(variances) - 1] = marg[2] ** 2

        # 将联合分布投影到的一维分布上
        index = 0
        for marg in groups[col]:
            y = marg[0].reshape(marg[3])  # 重新为分布修改状形，便于后续计算加权分布
            if marg[2].index(col) == 0:  # 如果属性对应的第一维，则按照第一维进行修改
                for i in range(lens):
                    w[index, i] = sum(y[i])
                variances[index] = marg[1] ** 2 * marg[3][1]
            else:
                for i in range(lens):
                    w[index, i] = sum(y[:, i])
                variances[index] = marg[1] ** 2 * marg[3][0]

            index += 1
        variance = 1.0 / np.sum(1.0 / variances)

        wgt_marg = np.zeros(lens)
        # 计算最小方差分布
        for i in range(lens):
            wgt_marg[i] = variance * sum(w[:, i] / variances)
        # print('wgt_marg', wgt_marg)
        # 存储方差最小化的一维分布
        Q = sparse.eye(wgt_marg.size)

        log1[col_index] = (Q, wgt_marg, np.sqrt(variance), log1[col_index][3])

        idx = 0
        for marg in log2:
            if col in marg[3] and len(marg[3]) == 1:
                log2[idx] = (log2[idx][0], wgt_marg, np.sqrt(variance), log2[idx][3])
            idx += 1

        # 基于最小方差分布，修正各联合分布,并存回噪音分布中
        index = 0
        for marg in groups[col]:
            avg_marg = np.zeros(marg[3])  # 满足一致性的噪音分布
            y = marg[0].reshape(marg[3])
            if marg[2].index(col) == 0:  # 如果属性对应的第一维，则按照第一维进行修改
                for i in range(lens):
                    avg_marg[i] = y[i] + (wgt_marg[i] - w[index, i]) / marg[3][1 - marg[2].index(col)]
            else:
                for i in range(lens):
                    avg_marg[:, i] = y[:, i] + (wgt_marg[i] - w[index, i]) / marg[3][1 - marg[2].index(col)]
            index += 1
            avg_marg = avg_marg.reshape(marg[3][0] * marg[3][1])
            log2[marg[4]] = (log2[marg[4]][0], avg_marg, marg[1], marg[2])
        col_index += 1
    return  total, log1, log2



def sub_data_combine(sub_data):
    # 最大团中每个团合成分布的对齐操作
    synth_data = sub_data[0]  # 用于记录所有团合并后的合成数据集分布
    synth_data.df = synth_data.df.sample(frac=1.0).reset_index(drop=True)

    for i in range(len(sub_data) - 1):
        inter_columns = list(set(list(synth_data.df)) & set(list(sub_data[i + 1].df)))
        if not inter_columns:  # 如果团和合成数据集没有交集 直接合并
            synth_data.df = pd.concat([synth_data.df, sub_data[i + 1].df], axis=1)
            synth_data.domain.attrs += sub_data[i + 1].domain.attrs
            synth_data.domain.shape += sub_data[i + 1].domain.shape
            synth_data.domain.config = dict(zip(synth_data.domain.attrs, synth_data.domain.shape))
        else:  # 如果有交集，先按照交集进行排序
            com_column = list((set(list(sub_data[i + 1].df)) ^ set(inter_columns)))
            synth_data.df.sort_values(inter_columns, inplace=True)
            synth_data.df = synth_data.df.reset_index(drop=True)  # 合成数据集根据交集列排序
            sub_data[i + 1].df.sort_values(inter_columns, inplace=True)
            sub_data[i + 1].df = sub_data[i + 1].df.reset_index(drop=True)
            sub_data[i + 1].df.drop(columns=inter_columns, inplace=True)  # 团数据根据交集列排序
            synth_data.df = pd.concat([synth_data.df, sub_data[i + 1].df], axis=1)
            com_shape = []
            for col in com_column:
                com_shape.append(sub_data[i + 1].domain.config[col])
            synth_data.domain.attrs += tuple(com_column)
            synth_data.domain.shape += tuple(com_shape)
            synth_data.domain.config = dict(zip(synth_data.domain.attrs, synth_data.domain.shape))
        synth_data.df = synth_data.df.sample(frac=1.0).reset_index(drop=True)
    return synth_data


# 将数据集投影至该团
def clique_data(data, clique):
    attr = []
    for col in data.domain:
        if col in clique:
            attr.append(col)
    data_clique = data.project(attr)
    return data_clique


def clique_log(data_clique, measurements2):
    edges = tuple(itertools.combinations(data_clique.domain.attrs, 2))
    edges +=tuple(itertools.combinations(data_clique.domain.attrs, 1))

    log2_clique = []
    # print('domain', data_clique.domain.attrs)
    for m in measurements2:
        if m[3] in edges:
            log2_clique.append(m)
    return log2_clique


def clique_log_v2(data_clique, measurement1, measurement2):
    edges = tuple(itertools.combinations(data_clique.domain.attrs, 2))
    edges1 = tuple(itertools.combinations(data_clique.domain.attrs, 1))

    log_clique = []

    for m in measurement2:
        if m[3] in edges:
            log_clique.append(m)

    for m in measurement1:
        if m[3] in edges1:
            log_clique.append(m)

    return log_clique



def check_if_fully_clique(max_cliques):
    fully_cliques = []
    for cliq in max_cliques:  # 循环遍历所有最大团，找出完备团
        fully_cliques.append(cliq)
    return fully_cliques


def measure(data, cliques, sigma_clique):
    measurements = []
    for proj, sigma in zip(cliques, sigma_clique):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma, proj))

    return measurements


def measure_single(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    print("sigma", sigma)
    print("real noise", sigma / wgt)
    return measurements


# 数据压缩算法
def compress_domain(data, measurements, total):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        if total * 0.0015 > 3 * sigma:
            sup = y >= 3 * sigma    # 阈值
        else:
            sup = y >= total * 0.0015
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append((Q, y, sigma, proj))
        else:  # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append((I2, y2, sigma, proj))
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


"""
以下是select_new以及会调用的函数
"""


def initialize(data):  # 全局变量Marginal_size,privacy,target的初始化
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xa = data.project(a).domain.shape
        xb = data.project(b).domain.shape
        Marginal_size.append(xa[0] * xb[0])  # Marginal_size记录与candidate数组对应的边际的大小
        privacy.append(0)
        target.append(0)  # 起初将所有的边际都设置为0，即未选择状态


def indif(data, rho1, cand_dom):  # 全局变量InDif的初始化
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    sigma = np.sqrt(1 / (2 * rho1))
    for a, b in candidates:
        indepent = []
        y = []
        xa = data.project(a).datavector()
        proba = np.array(xa) / len(data.df)
        xb = data.project(b).datavector()
        probb = np.array(xb) / len(data.df)
        x2 = data.project([a, b]).datavector()
        two_way = np.array(x2) / len(data.df)
        for i in range(len(proba)):
            for j in range(len(probb)):
                indepent.append(proba[i] * probb[j])
        for i in range(len(indepent)):
            y.append(indepent[i] - two_way[i])  # 这一步需要添加噪音

        #print(a, b, np.linalg.norm(y, 1) * len(data.df))
        cor = np.linalg.norm(y, 1) + np.random.normal(loc=0, scale=sigma * 4) / len(data.df)
        if cor < 0:
            cor = 0
        if cor >= (1 + (min(cand_dom[a], cand_dom[b]) - 2) / min(cand_dom[a], cand_dom[b])):
            cor = (1 + (min(cand_dom[a], cand_dom[b]) - 2) / min(cand_dom[a], cand_dom[b]))
        InDif.append(cor)
    return


def compute_graph_error(G_dif, total):
    # G_dif 是图模型的权重地点
    # total 是数据集大小
    candidates = list(itertools.combinations(data.domain.attrs, 2))  # 用于记录每个索引对应的边缘名称

    corre_error = 0.0
    noise_error = 0
    for i in range(len(target)):
        # 添加噪音时，如果桶数量较多，添加较少的噪音会在图模型训练后被置为0
        # 该现象或许是源于  logsumexp 函数的特性
        if target[i] == 1:
            noise_error += Marginal_size[i] / ((math.pi * privacy[i]) ** 0.5)
        else:
            corre_error += G_dif[candidates[i][0]][candidates[i][1]]['weight']

    corre_error = corre_error * total
    sumerror = corre_error + noise_error
    return sumerror


def compute_graph_error_v2(G_dif, targets, total):
    # G_dif 是图模型的权重地点
    # total 是数据集大小
    candidates = list(itertools.combinations(data.domain.attrs, 2))  # 用于记录每个索引对应的边缘名称

    corre_error = 0.0
    noise_error = 0
    for i in range(len(target)):
        if i in targets:
            noise_error += Marginal_size[i] / ((math.pi * privacy[i]) ** 0.5)
        else:
            corre_error += G_dif[candidates[i][0]][candidates[i][1]]['weight']

    corre_error = corre_error * total
    sumerror = corre_error + noise_error
    return sumerror

# error计算的是一个边际的误差，sum_error计算的是所有边际的总误差


def privacy_allocation(rho, temp_tar, sum_size):  # 隐私保护预算的分配
    for idx in temp_tar:
        privacy[idx] = rho * math.pow(Marginal_size[idx], 2/3) / sum_size
    return



# def select_new(data, rho1, rho2)
# rho1 用于添加噪音  indif函数中添加噪音 np.random.normal(loc=0, scale=sigma*4/len(data.df))
# 数据域大小不超过100W 对完备团的数据域进行检查


def select_new(data, rho1, rho2):  # 贪心算法函数
    cliques = []  # 由于原先的select函数是直接返回边际，所以这里也定义一个cliques用于返回所有选择的边际
    initialize(data)  # 调用函数，初始化
    cand_dom = {}
    for attr in data.domain.attrs:
        cand_dom[attr] = data.project(attr).domain.shape[0]
    indif(data, rho1, cand_dom)  # 初始化InDif的值
    candidates = list(itertools.combinations(data.domain.attrs, 2))     # 用于记录每个索引对应的边缘名称

    G_org = nx.Graph()      # 记录所有属性间相关性的无向图
    G_norm = nx.Graph()     # 根据属性域归一化相关性
    idx = 0
    for a, b in candidates:
        G_org.add_weighted_edges_from([(a, b, InDif[idx])])
        a_dom = cand_dom[a]
        b_dom = cand_dom[b]
        G_norm.add_weighted_edges_from([(a, b, InDif[idx] / (1 + (min(a_dom, b_dom) - 2) / min(a_dom, b_dom)))])
        idx += 1

    dif_dict = copy.deepcopy(G_org._adj)  # 记录org与cur间的差异以选择边缘和计算相关性误差

    G_cur = nx.Graph()  # 记录选择的属性间相关性以及推理其他属性的相关性

    final_errors = compute_graph_error(dif_dict, len(data.df))

    # 根据属性域归一化相关性
    norm_dif = copy.deepcopy(G_norm._adj)

    while True:  # 一直循环直到break
        all_errors = []  # 记录选择某一个边际后的总错误
        temp_target = []  # 记录每次循环会尝试选择的边际的index
        for i in range(len(target)):
            if target[i] == 0:  # 如果该边际未被选择
                # 以下3行用于计算：假如选择这个边际会得到的总误差
                target[i] = 1  # 把状态设为选择
                cliques.append(candidates[i])

                temp_cliques = []  # 临时的边缘集合
                temp_tar = []    # 临时选择索引

                # 针对当前的概率图模型，挑选需要度量的数据
                # 调用图模型的原因是为了在高隐私保护下，记录一维属性分布，即某个属性不与任何其他属性相连
                # 判断图模型的大小是否超过超过阈值，超过则放弃选择该边缘
                final_cliques = []
                model = GraphicalModel(data.domain, cliques)
                flag = 0
                for j in model.cliques:
                    if np.prod(data.domain.project(j).shape) > 1000000:
                        flag = 1
                    if len(j) > 1:
                        final_cliques += itertools.combinations(j, 2)
                        temp_cliques += itertools.combinations(j, 2)
                    else:
                        final_cliques.append(j)

                # 如果添加此条边缘使得团结构过大 则放弃选择此边缘
                if flag == 1:
                    target[i] = 0  # 重新把状态设为不选择
                    cliques.pop()
                    continue
                final_cliques = list(set(final_cliques))

                # 统计选择的边缘
                for marg in temp_cliques:
                    if target[candidates.index(marg)] != 2:
                        temp_tar.append(candidates.index(marg))
                    else:
                        flag = 1

                if flag == 1:
                    target[i] = 0  # 重新把状态设为不选择
                    cliques.pop()
                    continue

                # 计算总的属性域大小
                sum_size = 0
                k = 0
                for j in final_cliques:
                    sum_size += math.pow(data.domain.project(j).size(), 2/3)
                    k += 1

                # 根据当前的属性域大小计算噪音误差与相关性误差
                privacy_allocation(rho2, temp_tar, sum_size)  # 重新分配隐私预算


                # 创建临时的图结构 计算添加边缘后的相关性和总误差
                # temp_G_cur 将会通过类似dijkstra的方式推理当前结构对与所有边缘的分布确定程度
                # 创建图结构添加归一化的边缘
                temp_G_cur = nx.Graph()
                for idx in temp_tar:
                    att_1 = candidates[idx][0]
                    att_2 = candidates[idx][1]
                    noise = Marginal_size[idx] / ((math.pi * privacy[idx]) ** 0.5) / len(data.df)
                    wgt = norm_dif[att_1][att_2]['weight'] - noise
                    if wgt < 0:
                        wgt = 0
                    temp_G_cur.add_weighted_edges_from([(att_1, att_2, wgt)])

                # 使用类似于dijkstra方式推理当前结构对与所有边缘的分布确定程度
                # 对所有节点计算其他节点的相关性
                for k in temp_G_cur._adj.keys():
                    corre, paths = multi_source_correlation(temp_G_cur, {k})
                    for u, e in corre.items():
                        temp_G_cur.add_weighted_edges_from([(k, u, e)])

                # 图结构当前的差异
                dif_dict = update_dif(temp_G_cur._adj, G_norm._adj)
                for a, b in candidates:
                    a_dom = cand_dom[a]
                    b_dom = cand_dom[b]
                    dif_dict[a][b]['weight'] *= (1 + (min(a_dom, b_dom) - 2) / min(a_dom, b_dom))

                # 删除当前的测试属性
                cliques.pop()


                # 用s记录选择这个边际得到的总误差
                s = compute_graph_error_v2(dif_dict, temp_tar, len(data.df))
                all_errors.append(s)
                target[i] = 0  # 重新把状态设为不选择
                temp_target.append(temp_tar)

        et = list(zip(all_errors, temp_target))  # et是一个列表，列表里元素的模式为（如果选择边际得到的总误差，边际的index）


        sorted_et = sorted(et)  # 根据总误差大小从小到大排序
        if (sorted_et == []):
            break
        update_errors = sorted_et[0][0]  # 用update_errors接收最小的总误差

        if update_errors >= final_errors:  # 根据伪代码，如果加入边际后反而使得误差变大了那就break
            break
        else:
            cliques = []
            for i in range(len(sorted_et[0][1])):
                target[sorted_et[0][1][i]] = 1  # 选择边际
                cliques.append(candidates[sorted_et[0][1][i]])  # 加入cliques中
            # 检验
            model = GraphicalModel(data.domain, cliques)
            fully_cliques = check_if_fully_clique(model.cliques)

            flag = True
            for clique in fully_cliques:
                if np.prod(data.domain.project(clique).shape) > 1000000:
                    flag = False
                    break
            if flag == True:
                final_errors = update_errors  # 更新final_errors
                for i in range(len(sorted_et[0][1])):
                    G_cur.add_weighted_edges_from([(candidates[sorted_et[0][1][i]][0], candidates[sorted_et[0][1][i]][1], norm_dif[candidates[sorted_et[0][1][i]][0]][candidates[sorted_et[0][1][i]][1]]['weight'])])
            else:
                for i in range(len(sorted_et[0][1])):
                    target[sorted_et[0][1][i]] = 2  # 放弃选择
                    cliques.pop()  # 删除刚刚添加的边际

    # 针对当前的概率图模型，挑选需要度量的数据
    # 调用图模型的原因是为了在高隐私保护下，记录一维属性分布，即某个属性不与任何其他属性相连
    final_cliques = []
    model = GraphicalModel(data.domain, cliques)
    for i in model.cliques:
        if len(i) > 1:
            final_cliques += itertools.combinations(i, 2)
        else:
            final_cliques.append(i)
    final_cliques = list(set(final_cliques))

    print('number of cliques', len(cliques))
    print(cliques)
    return final_cliques


def default_params():
    """
    Return default parameters to run this program
    :returns: a dictionary of default parameter settings for each command line argument
    """

    params = {}
    params['dataset'] = '../datasets/titanic.csv'
    params['domain'] = '../datasets/titanic-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 1000000000
    params['save'] = 'syn.csv'
    return params


if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--save', type=str, help='path to save synthetic data')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    time_start = time.time()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    workload3 = list(itertools.combinations(data.domain, 3))
    workload3 = [cl for cl in workload3 if data.domain.size(cl) <= args.max_cells]
    workload4 = list(itertools.combinations(data.domain, 4))
    workload4 = [cl for cl in workload4 if data.domain.size(cl) <= args.max_cells]

    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]


    print('epsilon', args.epsilon, 'delta', args.delta)

    synth, cliques = MST(data, args.epsilon, args.delta)

    time_end = time.time()
    print('totally cost', time_end - time_start)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    '''
    # 计算一致性处理前的一维分布误差
    errors = []
    cliques = [(col,) for col in data.domain]
    for proj in cliques:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        print(proj, e)
        errors.append(e)
    print('Average 1-way Error : ', np.mean(errors))
    '''

    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print('Average 2-way Error: ', np.mean(errors))

    errors = []
    for proj in workload3:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print('Average 3-way Error : ', np.mean(errors))


    errors = []
    for proj in workload4:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print('Average 4-way Error : ', np.mean(errors))
