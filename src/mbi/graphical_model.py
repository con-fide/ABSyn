import numpy as np
from mbi import Domain, Dataset, CliqueVector
from mbi.junction_tree import JunctionTree
from functools import reduce
import pickle
import networkx as nx
import itertools
import pandas as pd


class GraphicalModel:
    def __init__(self, domain, cliques, total=1.0, elimination_order=None):
        """ Constructor for a GraphicalModel

        :param domain: a Domain object
        :param total: the normalization constant for the distribution
        :param cliques: a list of cliques (not necessarilly maximal cliques)
            - each clique is a subset of attributes, represented as a tuple or list
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.  
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.total = total
        tree = JunctionTree(domain, cliques, elimination_order)
        self.junction_tree = tree
        self.cliques = tree.maximal_cliques()   # 属性的最大团集合
        self.message_order = tree.mp_order()    # 消息传递的路径
        self.sep_axes = tree.separator_axes()   # 联结树中消息传递的边
        self.neighbors = tree.neighbors()
        self.elimination_order = tree.elimination_order

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size * 8 > 4 * 10 ** 9:
            print('cliques', cliques)
            print('max cliques haha', self.cliques)
            import warnings
            message = 'Size of parameter vector is %.2f GB. ' % (self.size * 8 / 10 ** 9)
            message += 'Consider removing some measurements or finding a better elimination order'
            warnings.warn(message)

    @staticmethod
    def save(model, path):
        pickle.dump(model, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

    def project(self, attrs):       # 将数据分布投影到属性的子集上
        """ Project the distribution onto a subset of attributes.
            I.e., compute the marginal of the distribution

        :param attrs: a subset of attributes in the domain, represented as a list or tuple
        :return: a Factor object representing the marginal distribution
        """
        # use precalculated marginals if possible
        if type(attrs) is list:
            attrs = tuple(attrs)
        if hasattr(self, 'marginals'):
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim = self.domain.invert(attrs)    # elim为包含于domain而不在attrs中的属性
        elim_order = greedy_order(self.domain, self.cliques, elim)      # 贪心算法得到消除属性的顺序
        pots = list(self.potentials.values())
        ans = variable_elimination_logspace(pots, elim_order, self.total)
        return ans.project(attrs)

    def krondot(self, matrices):
        """ Compute the answer to the set of queries Q1 x Q2 X ... x Qd, where 
            Qi is a query matrix on the ith attribute and "x" is the Kronecker product
        This may be more efficient than computing a supporting marginal then multiplying that by Q.
        In particular, if each Qi has only a few rows.
        
        :param matrices: a list of matrices for each attribute in the domain
        :return: the vector of query answers
        """
        assert all(M.shape[1] == n for M, n in zip(matrices, self.domain.shape)), \
            'matrices must conform to the shape of the domain'
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        Factor = type(factors[0])  # infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(['%s-answer' % attr, attr], Q.shape)
            factors.append(Factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(['%s-answer' % a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    def calculate_many_marginals(self, projections):
        """ Calculates marginals for all the projections in the list using
            Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)
    
        This method may be faster than calling project many times
        
        :param projections: a list of projections, where 
            each projection is a subset of attributes (represented as a list or tuple)
        :return: a list of marginals, where each marginal is represented as a Factor
        """

        self.marginals = self.belief_propagation(self.potentials)
        sep = self.sep_axes
        neighbors = self.neighbors
        # first calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(self.junction_tree.tree, weight=False)
        results = {}
        for Ci, Cj in sorted(itertools.combinations(self.cliques, 2), key=lambda X: dist[X[0]][X[1]]):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {self.domain.canonical(key[0] + key[1]): results[key] for key in results}

        answers = {}
        for proj in projections:
            for attr in results:
                if set(proj) <= set(attr):
                    answers[proj] = results[attr].project(proj)
                    break
            if proj not in answers:
                # just use variable elimination
                answers[proj] = self.project(proj)

        return answers

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def datavector_vec(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector_vec(flatten) * wgt * self.total

    def belief_propagation(self, potentials, logZ=False):   # 联结树m个节点只需要2(m-1)次消息传递
        """ Compute the marginals of the graphical model with given parameters
        
        Note this is an efficient, numerically stable implementation of belief propagation
    
        :param potentials: the (log-space) parameters of the graphical model
        :param logZ: flag to return logZ instead of marginals
        :return marginals: the marginals of the graphical model
        """
        beliefs = {cl: potentials[cl].copy() for cl in potentials}      #每个最大团 有一个势函数
        messages = {}
        for i, j in self.message_order:     # 按照沿着信念传播的拓扑排序
            sep = beliefs[i].domain.invert(self.sep_axes[(i, j)])   # 关于团i的势函数, sep记录在团i而不在j中的属性集合
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)]     # 如果有相反方向的消息传播，即减去对应的势
            else:
                tau = beliefs[i]
            messages[(i, j)] = tau.logsumexp(sep)   # tau的计算考虑交集为sep，将势函数投影到对应的值上
            beliefs[j] += messages[(i, j)]      # 团i中势函数的改变传递到团j中  （相同的值域+相同的value）
        cl = self.cliques[0]
        if logZ: return beliefs[cl].logsumexp()
        # 通过logsumexp方法增强数值稳定性
        logZ = beliefs[cl].logsumexp()
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl])
        return CliqueVector(beliefs)

    def mle(self, marginals):
        """ Compute the model parameters from the given marginals

        :param marginals: target marginals of the distribution
        :param: the potentials of the graphical model with the given marginals
        """
        potentials = {}
        variables = set()
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            # factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            potentials[cl] = marginals[cl].log() - marginals[cl].project(new).log()
        return CliqueVector(potentials)

    def fit(self, data):
        from mbi import Factor
        assert data.domain.contains(self.domain), 'model domain not compatible with data domain'
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def synthetic_data(self, rows=None):
        """ Generate synthetic tabular data from the distribution """
        total = int(self.total) if rows is None else rows  # 生成total条记录
        cols = self.domain.attrs  # cols记录属性
        data = np.zeros((total, len(cols)), dtype=int)  # 初始化数据集
        df = pd.DataFrame(data, columns=cols)   # 生成数据格式
        cliques = [set(cl) for cl in self.cliques]

        def synthetic_col(counts, total):   # 合成数据集中添加属性
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)   # 整数部分
            # print('count', counts)
            # print('small', integ)
            extra = total - integ.sum()
            # if extra > 0:
            #    o = np.argsort(frac)
            #    integ[o[-extra:]] += 1
            if extra > 0:

                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1]    # 根据模型的变量消除顺序生成记录
        col = order[0]
        marg = self.project([col]).datavector(flatten=False)       # marg投影到col上
        df.loc[:, col] = synthetic_col(marg, total) # 按照第一个属性的分布生成记录
        used = {col}    # 保存已统计过的属性

        # 逐步按照数据分布填充每个属性的对应分布
        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]  # list用于保存包含属性col的最大团
            # print('* relevant', *relevant)
            relevant = used.intersection(set.union(*relevant))  # relevant与当前度量过的属性的交集
            # print('intersection relevant', relevant)
            proj = tuple(relevant)
            used.add(col)       # 添加当前要度量的属性col
            marg = self.project(proj + (col,)).datavector(flatten=False)    # 将数据分布投影到当前属性，以及与它处在同一个团中且已经度量过的属性

            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj)).apply(foo)  # 如果有交集，分组统计度量过的属性的分布
            else:
                df[col] = synthetic_col(marg, df.shape[0])  # 如果当前构造的属性和其他属性没有交集，直接调用synthetic_col函数
        return Dataset(df, self.domain)

    def synthetic_datavector(self, rows=None):
        """ Generate synthetic tabular data from the distribution """
        total = int(self.total) if rows is None else rows  # 生成total条记录
        cols = self.domain.attrs  # cols记录属性
        data = np.zeros((total, len(cols)), dtype=int)  # 初始化数据集
        df = pd.DataFrame(data, columns=cols)   # 生成数据格式
        cliques = [set(cl) for cl in self.cliques]

        def synthetic_col(counts, total):   # 合成数据集中添加属性
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)   # 整数部分
            # print('count', counts)
            # print('small', integ)
            extra = total - integ.sum()
            # if extra > 0:
            #    o = np.argsort(frac)
            #    integ[o[-extra:]] += 1
            if extra > 0:
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1]    # 根据模型的变量消除顺序生成记录
        col = order[0]
        marg = self.project([col]).datavector_vec(flatten=False)       # marg投影到col上  一致性处理后，找一个团通过datavector投影
        df.loc[:, col] = synthetic_col(marg, total) # 按照第一个属性的分布生成记录
        used = {col}    # 保存已统计过的属性

        # 逐步按照数据分布填充每个属性的对应分布
        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]  # relevant用于保存包含属性col的团
            # print('* relevant', *relevant)
            relevant = used.intersection(set.union(*relevant))  # relevant与当前度量过的属性的交集，查找是否团中同时包含度量过的属性，以及该边缘
            # print('intersection relevant', relevant)
            proj = tuple(relevant)
            used.add(col)       # 添加当前要度量的属性col
            marg = self.project(proj + (col,)).datavector_vec(flatten=False)    # 将数据分布投影到当前属性，以及与它处在同一个团中且已经度量过的属性

            '''
            relevant包含了与当前生成的属性col以及和col处于相同团的已经生成过的属性
            marg表示relevant包含属性的联合分布
            '''

            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj)).apply(foo)  # 如果有交集，分组统计度量过的属性的分布
            else:
                df[col] = synthetic_col(marg, df.shape[0])  # 如果当前构造的属性和其他属性没有交集，直接调用synthetic_col函数
        return Dataset(df, self.domain)

def variable_elimination_logspace(potentials, elim, total):     # 给定势函数  变量消除顺序和样本总数  对数据进行投影
    """ run variable elimination on a list of **logspace** factors """
    k = len(potentials)
    # print('number of potentials', k)
    # print('elimination order', elim)
    psi = dict(zip(range(k), potentials))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]     # psi2 为包含属性z的团psi[i]的集合(除去本次投影的属性)
        phi = reduce(lambda x, y: x + y, psi2, 0)
        # print('phi value', phi.values)
        tau = phi.logsumexp([z])
        psi[k] = tau
        k += 1
    ans = reduce(lambda x, y: x + y, psi.values(), 0)
    return (ans - ans.logsumexp() + np.log(total)).exp()    # 将势转换为实际数据分布

def variable_elimination(factors, elim):
    """ run variable elimination on a list of (non-logspace) factors """
    k = len(factors)
    psi = dict(zip(range(k), factors))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x * y, psi2, 1)
        tau = phi.sum([z])
        psi[k] = tau
        k += 1
    return reduce(lambda x, y: x * y, psi.values(), 1)


def greedy_order(domain, cliques, elim):
    order = []
    unmarked = set(elim)
    cliques = set(cliques)
    total_cost = 0
    for k in range(len(elim)):
        cost = {}
        for a in unmarked:
            # all cliques that have a
            neighbors = list(filter(lambda cl: a in cl, cliques))
            # variables in this "super-clique"
            variables = tuple(set.union(set(), *map(set, neighbors)))
            # domain for the resulting factor
            newdom = domain.project(variables)
            # cost of removing a
            cost[a] = newdom.size()

        # find the best variable to eliminate
        a = min(cost, key=lambda a: cost[a])

        # do some cleanup
        order.append(a)
        unmarked.remove(a)
        neighbors = list(filter(lambda cl: a in cl, cliques))
        variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order
