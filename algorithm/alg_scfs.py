import csv
import pandas as pd
import numpy as np
import random as r
import networkx as nx
import math
import time
from igraph import Graph
import multiprocessing as mp
from multiprocessing import freeze_support

# 依据 Fig.2 给的算法伪代码逻辑，采用递归的方式来实现
def alg_SCFS(A_rm: np.ndarray, paths_obs: np.ndarray, queue=None):
    '''
        - 在树型拓扑中，对于每条链路 (s, d)，其可借由它的末端节点 link_d 来唯一指代，而不引起歧义
        - 正常/拥塞状态：0/1，注意与论文中相反
        - 链路/路径变量: X/Y，注意与论文中相反

        * Python 内部函数以及变量深拷贝参见
            = https://blog.51cto.com/u_14246112/3157550
            = https://blog.csdn.net/DeniuHe/article/details/77370112
    '''
    from copy import deepcopy

    W_s = np.zeros(paths_obs.shape[0], dtype=object)
    for idx, path_obs in enumerate(paths_obs):
        def recurse(k):  # 内部函数：在树型拓扑中递归地诊断链路 link_k
            nonlocal X, Y, W, R
            if k in R:
                X[k - 1] = Y[np.where(R == k)[0][0]]
            else:
                childNodeSet = d(k)
                for j in childNodeSet:
                    recurse(j)
                if k != 0:
                    X[k - 1] = min([X[j - 1] for j in childNodeSet])
                for j in childNodeSet:
                    if X[j - 1] == 1 and X[k - 1] == 0:
                        W.append(j)
                    if all(X):
                        W.append(1)

        def d(k):  # 内部函数：在树型拓扑中获取节点 k 的 child node 集合
            nonlocal linkSet_in_eachPath
            if k == 0:
                childNode_set = np.array([1])
            else:
                childNode_set = np.array(
                    [linkSet_in_eachPath[i][np.where(linkSet_in_eachPath[i] == k)[0][0] + 1] for i in
                     range(linkSet_in_eachPath.shape[0]) if
                     np.where(linkSet_in_eachPath[i] == k)[0].shape[0] != 0])
                childNode_set = np.unique(childNode_set)
            return childNode_set

        _, num_links = A_rm.shape  # 获取链路数量

        Y = deepcopy(path_obs)  # 路径状态观测值（已给定）；可以不进行 deepcopy 操作，因为整个程序不涉及修改 Y 的值；仅做提示作用
        X = np.zeros(num_links, dtype=int)  # 链路状态值（待估计）
        R = np.array([i + 1 for i in range(A_rm.shape[1]) if np.where(A_rm[:, i] == 1)[0].shape[0] == 1],
                     dtype=int)  # 叶节点
        linkSet_in_eachPath = np.array([np.where(A_rm[i] == 1)[0] + 1 for i in range(A_rm.shape[0])],
                                       dtype=object)  # 每条路径上的链路集

        W = []

        recurse(0)
        W_s[idx] = np.unique(np.array(W))
    if queue is None:
        return W_s
    else:
        queue.put(W_s)

def am2rm(adjacency):
    # from adjacency to routine matrix
    # 1. from adjacency to the list of edges
    edges = []
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i][j] == 1:
                edges.append((i, j))
    print(edges)
    # 2. from the list of edges to routine matrix
    # 2.1 get_leaf_nodes
    return


def rm2am(routine_matrix):
    # from routine_matrix to adjacency_matrix
    # 1. from routine_matrix to edges
    each_path = get_each_path(routine_matrix)
    edges = []
    for i in range(each_path.shape[0]):
        for j in range(len(each_path[i]) - 1):
            edge = (each_path[i][j + 1], each_path[i][j])
            if edge not in edges:
                edges.append(edge)
    # 2. from edgelist to adjacency
    G = nx.from_edgelist(edges)
    matrix = nx.to_numpy_matrix(G)

    return matrix


def get_each_path(A_rm):
    # 倒序
    nNum = np.zeros(A_rm.shape[1], dtype=int)
    for i in range(A_rm.shape[1]):
        cnt = 0
        for j in range(A_rm.shape[0]):
            if A_rm[j][i] == 1:
                cnt += 1
        nNum[i] = cnt
    sort_nNum = sorted(nNum, reverse=True)

    def sort(num):
        return sort_nNum.index(nNum[num - 1])

    each_path = np.zeros(A_rm.shape[0], dtype=list)

    for i in range(A_rm.shape[0]):
        path = []
        for j in range(A_rm.shape[1]):
            if A_rm[i][j] == 1:
                path.append(j + 1)
        each_path[i] = sorted(path, key=sort, reverse=True)

    return each_path


def rm2Graph(fname):
    data = pd.read_csv(fname)
    rm = np.array(data)
    matrix = rm2am(rm)
    G = nx.from_numpy_matrix(matrix)
    return G


def csv2Graph(fname):
    data = pd.read_csv(fname)
    g = nx.from_pandas_edgelist(data, source="Id", target="Target")
    return g


def gml2Graph(fname):
    g = Graph.Read_GML(fname)
    del g.vs["label"]
    return g


def show_topo(fName, fFormat):
    try:
        if fFormat == "Routine_Matrix(*.csv)" and fName.endswith('.csv'):
            G = rm2Graph(fName)
        elif fFormat == "Adjacency_Matrix(*.csv)" and fName.endswith('.csv'):
            G = csv2Graph(fName)
        elif fFormat == "Topology_Zoo(*.gml)" and fName.endswith('.gml'):
            G = gml2Graph(fName)
        else:
            # sg.popup_notify(f'[-] Error: 格式错误...')
            return 0
        return G
    except Exception as e:
        # sg.popup_notify(e)
        return 0


def csv2edgeList(csvfile):
    edges_df = pd.read_csv("C:\\Users\\Eric_Teng\\Desktop\\%s" % csvfile)[['Source', 'Target']]
    # print(edges_df)
    edges_tuple = edges_df.apply(lambda x: tuple(x), axis=1).values.tolist()
    print(edges_tuple)
    return edges_tuple


def random_links_obs(length: int):
    n = 1000
    prob = round(r.choice(np.arange(0.005, 0.305, 0.005)), 3)
    # print("链路拥塞概率为:\n", prob)
    multi_links_obs = 1 * (np.random.rand(n, length) < prob)
    return multi_links_obs


def paths_stat(A_rm, links_stat):
    times = links_stat.shape[0]
    paths_num = A_rm.shape[0]

    paths_obs = np.zeros(times, dtype=np.ndarray)
    for i in range(times):  # 时间轮数times
        path_obs = np.zeros(paths_num, dtype=int)
        for j in range(paths_num):  # 路径数
            tmp = links_stat[i] * A_rm[j]
            if tmp.any():
                path_obs[j] = 1
        paths_obs[i] = path_obs
    return paths_obs


def paths_stat_single(A_rm, links_stat):
    paths_num = A_rm.shape[0]
    path_obs = np.zeros(paths_num, dtype=int)
    for i in range(paths_num):  # 路径数
        tmp = links_stat * A_rm[i]
        # print(tmp)
        # print(tmp.any())
        if tmp.any():
            path_obs[i] = 1
        # print(path_obs)
    # print(paths_obs)
    return path_obs


def get_each_path(A_rm):
    nNum = np.zeros(A_rm.shape[1], dtype=int)
    for i in range(A_rm.shape[1]):
        cnt = 0
        for j in range(A_rm.shape[0]):
            if A_rm[j][i] == 1:
                cnt += 1
        nNum[i] = cnt
    sort_nNum = sorted(nNum, reverse=True)

    def sort(num):
        return sort_nNum.index(nNum[num - 1])

    each_path = np.zeros(A_rm.shape[0], dtype=list)
    for i in range(A_rm.shape[0]):
        path = []
        for j in range(A_rm.shape[1]):
            if A_rm[i][j] == 1:
                path.append(j + 1)
        each_path[i] = sorted(path, key=sort, reverse=True)
    return each_path


def edges_tv(tree_vector: np.ndarray):
    edges = []
    for i in range(tree_vector.shape[0]):
        start = int(tree_vector[i])
        end = i + 1
        edge = (start, end)
        edges.append(edge)
    return edges


def tree_vector_rm(A_rm: np.ndarray):
    each_path = get_each_path(A_rm)

    tree_vector = np.zeros(A_rm.shape[1], dtype=int)
    for i in range(each_path.shape[0]):
        path = each_path[i]
        for j in range(1, len(path)):
            tree_vector[path[j - 1] - 1] = path[j]
    return tree_vector


def get_leaf_nodes(A_rm):
    each_path = get_each_path(A_rm)
    leaf_nodes = np.zeros(each_path.shape[0], dtype=int)
    for i in range(each_path.shape[0]):
        leaf_nodes[i] = each_path[i][0]
    return leaf_nodes


def SCFS(A_rm: np.ndarray, path_obs: np.ndarray):
    result = np.zeros(0)
    for i in range(path_obs.shape[0]):  # 轮次为时间状态数
        y = path_obs[i]
        link_health = []
        link_states_estimated = [0] * A_rm.shape[1]
        for j in range(y.shape[0]):
            if int(y[j]) == 0:  # 该条路径正常
                for k in range(A_rm.shape[1]):
                    if int(A_rm[j][k]) == 1:
                        if k not in link_health:
                            link_health.append(k)
        for j in range(y.shape[0]):
            if int(y[j]) == 1:  # 该条路径拥塞
                for k in range(A_rm.shape[1]):
                    if int(A_rm[j][k]) == 1:
                        if link_states_estimated[k] == 1:
                            break
                        if k not in link_health:
                            link_states_estimated[k] = 1
                            break
        result = np.append(result, link_states_estimated)
    result = result.reshape(path_obs.shape[0], -1).transpose()
    result = result.astype(np.int64)
    return result


def detection(links_state: np.ndarray, links_state_inferred: np.ndarray, multiple: bool):
    # Params: TP, FP, TN, FN
    # Middle: PPV(Precision), TPR(Sensitivity, Recall)
    # Return: F1-Score(From 0 - 1, which means worst or best)
    if multiple:
        DR = np.zeros(links_state.shape[0], dtype=float)
        FPR = np.zeros(links_state.shape[0], dtype=float)
        F1 = np.zeros(links_state.shape[0], dtype=float)
        for i in range(links_state.shape[0]):
            if np.array_equal(links_state[i], links_state_inferred[i]):
                DR[i] = 1
                FPR[i] = 0
                F1[i] = 1

            else:
                FC = np.where(links_state[i] == 1)[0]  # 真实拥塞链路
                XC = np.where(links_state_inferred[i] == 1)[0]  # 模拟拥塞链路
                U = np.array(list(range(links_state[i].shape[0])))  # 全集取反
                FU = np.setdiff1d(U, FC)  # 真实正常链路
                XU = np.setdiff1d(U, XC)  # 模拟正常链路
                TP = np.intersect1d(FC, XC)
                FP = np.intersect1d(FU, XC)
                TN = np.intersect1d(FU, XU)
                FN = np.intersect1d(FC, XU)
                if TP.shape[0] != 0:
                    PPV = round(TP.shape[0] / (TP.shape[0] + FP.shape[0]), 2)
                    TPV = round(TP.shape[0] / (TP.shape[0] + FN.shape[0]), 2)
                    FPR[i] = round(FP.shape[0] / (FP.shape[0] + TN.shape[0]), 2)
                    F1[i] = round((2 * PPV * TPV) / (PPV + TPV), 2)
                    DR[i] = TPV
                else:
                    DR[i] = 0
                    FPR[i] = 1
                    F1[i] = 0
                # if FPR[i] == 1:
                #     print(links_state[i], links_state_inferred[i])
    else:
        if np.array_equal(links_state, links_state_inferred):
            DR = 1
            FPR = 0
            F1 = 1
        else:
            FC = np.where(links_state == 1)[0]  # 真实拥塞链路
            XC = np.where(links_state_inferred == 1)[0]  # 模拟拥塞链路
            U = np.array(list(range(links_state.shape[0])))  # 全集取反
            FU = np.setdiff1d(U, FC)  # 真实正常链路
            XU = np.setdiff1d(U, XC)  # 模拟正常链路
            TP = np.intersect1d(FC, XC)
            FP = np.intersect1d(FU, XC)
            TN = np.intersect1d(FU, XU)
            FN = np.intersect1d(FC, XU)
            if TP.shape[0] != 0:
                PPV = round(TP.shape[0] / (TP.shape[0] + FP.shape[0]), 2)
                TPV = round(TP.shape[0] / (FC.shape[0]), 2)
                FPR = round(FP.shape[0] / (XC.shape[0]), 2)
                F1 = round((2 * PPV * TPV) / (PPV + TPV), 2)
                DR = TPV
            else:
                DR = 0
                FPR = 1
                F1 = 0

    return DR, FPR, F1


def table_list(links_state, links_state_inferred):
    good_table_list = []
    bad_table_list = []
    DR, FPR, F1 = detection(links_state, links_state_inferred, multiple=True)
    # print(f'DR\n{DR}\nFPR\n{FPR}\nF1\n{F1}')
    bName = './topology/bad_performance.csv'
    bcsv = open(bName, 'w', newline='')
    bWriter = csv.writer(bcsv)
    header = ['DR', 'FPR', 'F1']
    bWriter.writerow(header)

    for i in range(DR.shape[0]):
        if DR[i] == 1 and FPR[i] == 0 and F1[i] == 1:
            good_table_list.append(f'DR:  1.00,  FPR:  0.00,  F1:  1.00')
        else:
            bWriter.writerow([DR[i], FPR[i], F1[i]])
            # bad_table_list.append('DR:  %0.2f,  FPR:  %0.2f,  F1:  %0.2f' % (DR[i], FPR[i], F1[i]))
    # print(len(bad_table_list))
    bcsv.close()
    bad_df = pd.read_csv(bName).sort_values(by=['F1'], ascending=False).reset_index(drop=True)
    bad_table_list = ['DR:  %0.2f,  FPR:  %0.2f,  F1:  %0.2f' % (bad_df['DR'][i], bad_df['FPR'][i], bad_df['F1'][i])
                      for i in range(len(bad_df['F1']))]
    return good_table_list, bad_table_list


def traversal_through_F(A_rm, FNum, q=None):
    # specify the num of |F|
    linkNum = A_rm.shape[1]
    multi_conditions = np.zeros(0)
    for i in range(linkNum - FNum + 1):
        cnt = 0
        while True:
            single_condition = np.zeros(linkNum, dtype=int)
            single_condition[i] = 1
            for j in r.sample(range(i + 1, linkNum), FNum - 1):
                single_condition[j] = 1
            if multi_conditions.size == 0:
                multi_conditions = np.append(multi_conditions, single_condition).astype(int).reshape(-1, linkNum)
                cnt += 1
            else:
                if not np.any(np.all(single_condition == multi_conditions, axis=1)):
                    multi_conditions = np.append(multi_conditions, single_condition).astype(int).reshape(-1, linkNum)
                    cnt += 1
            if cnt == int(
                    math.factorial(linkNum - i - 1) / (math.factorial(FNum - 1) * math.factorial(linkNum - i - FNum))):
                break
    if q is None:
        return multi_conditions
    else:
        q.put(multi_conditions)


def paths_SCFS(A_rm, links_state, q):
    paths_obs = paths_stat(A_rm, links_state)
    links_state_inferred = SCFS(A_rm, paths_obs).transpose()
    # print(links_state_inferred.shape)
    q.put(links_state_inferred)


def traversal_through_Y(A_rm, YNum):
    # specify the num of Y
    if True:
        print("Timer: traversal_through_Y")
        multi_conditions_Y = np.zeros(0)
        freeze_support()
        q = mp.Queue()
        jobs = []
        results = np.zeros(0)
        start_time = time.time()
        for i in range(YNum, A_rm.shape[1]):
            p = mp.Process(target=traversal_through_F, args=(A_rm, i, q))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join(timeout=3)
        for j in jobs:
            results = np.append(results, q.get()).reshape(-1, A_rm.shape[1])

        print(f'State 1: {round(time.time() - start_time, 2)}s')
        q2 = mp.Queue()
        jobs2 = []
        results2 = np.zeros(0)
        processingNum = 3
        links_fragments = np.zeros(processingNum, dtype=np.ndarray)
        FRAGMENT_SIZE = int(round(results.shape[0] / processingNum, 0))
        for i in range(processingNum):
            links_fragments[i] = results[i * FRAGMENT_SIZE:(i + 1) * FRAGMENT_SIZE]
        for i in range(processingNum):
            p = mp.Process(target=paths_SCFS, args=(A_rm, links_fragments[i], q2))
            jobs2.append(p)
            p.start()
        for p in jobs2:
            p.join(timeout=2)
        print(f'State 2: {round(time.time() - start_time, 2)}s')
        for j in jobs2:
            results2 = np.append(results2, q2.get()).reshape(-1, A_rm.shape[1])
        print(f'State 3: {round(time.time() - start_time, 2)}s')
        for j in range(results2.shape[0]):
            if np.where(results2[j] == 1)[0].shape[0] == YNum:
                multi_conditions_Y = np.append(multi_conditions_Y, results[j]).astype(int).reshape(-1, A_rm.shape[1])
        print(f'In total: {round(time.time() - start_time, 2)}s')
    return multi_conditions_Y
