# Python Modelling API
# By Chen, Ziyuan
# Build 220213
#
#
# Chapter   Prefix        Description                          Notes
#    4      stat_         Mathematical Statistics
#    7      plot_         Plotting (2D/3D)
#           interp_       Interpolation (2D/3D)
#           curvefit_     Curve fitting (2D/3D)                "covmat" for np.ndarray, "curvefit_2d['score']" [COMING SOON]
#    8      odeivp_       ODE initial value problem solver
#    9      normalize_    Data normalizer
#           cem_          Comprehensive Evaluation Method
#   10      graph_        Graph algorithms
#           network_      Network algorithms (with capacity)   Interface for "capacity" and "demand" unstable
#   11      classify_     Classifying algorithms
#           multivar_     Multivariable analysis
#           cluster_      Cluster analysis
#   12      regression_   Regression algirithms
#   13                    Difference equation solver           [COMING SOON]
#   14      fuzzy_        Fuzzy mathematics algirithms         "fuzzy_cem" [COMING SOON]
#   15      graymodel_    Gray model prediction
#   17      intel_        Intelligent optimizers               Universality not guaranteed
#   18      timeseries_   Time series analysis
#   19      svm_          Support vector machine
#
#
# More Issues
#   - Switch for normalization (minmax, maxabs, standard, vectornorm, ...)
#   - Versatile interface supporting both "np.ndarray" and "pd.DataFrame"
#   - WARNING: Input integrity checks are left out in most cases
#   - Chapters 5, 6 OMITTED: For linear/non-linear/integer programming, use API from "cvxpy."
#   - Chapter 16 OMITTED: Monte Carlo simulation is unsuitable for unified APIs. 



import numpy as np
import sympy as sp
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

default_figsize = (5, 5)
default_dpi = 100



# Chapter 4

def stat_t_test(data, target_mean, alpha=0.05, side=1):
    import scipy.stats as ss
    from statsmodels.stats.weightstats import ztest
    t_stat, p = ztest(data, value=target_mean)
    t_boundary = ss.t.ppf(alpha/side, data.size-1)
    return (t_boundary, t_stat, p)

def stat_chisquare_test_discrete(data, target_freqs, alpha=0.05):
    import scipy.stats as ss
    chi_stat = (data**2/(data.sum()*target_freqs)).sum() - data.sum()
    chi_boundary = ss.chi2.ppf(1-alpha, data.size-1)
    return chi_boundary, chi_stat

def stat_chisquare_test_continuous_norm(data, alpha=0.05, bins=5):
    import scipy.stats as ss
    data_hist = plt.hist(data, bins=bins)
    data_counts = data_hist[0]
    data_bounds = data_hist[1]
    target_freqs = np.diff(ss.norm.cdf(data_bounds, data.mean(), data.std()))
    target_freqs /= target_freqs.sum()
    chi_stat = (data_counts**2/(data_counts.sum()*target_freqs)).sum() - data_counts.sum()
    chi_boundary = ss.chi2.ppf(1-alpha, bins-3)
    return chi_boundary, chi_stat

def stat_ks_test(data, func='norm'):
    import scipy.stats as ss
    return ss.kstest(data, func, (data.mean(), data.std()))

def stat_anova(data, categoryA, categoryB):
    import statsmodels.api as sm
    if not isinstance(x2, type(None)): 
        model = sm.formula.ols("y~C(x1)*C(x2)", {'x1':categoryA, 'x2':categoryB, 'y':data}).fit()
    else: model = sm.formula.ols("y~C(x)", {'x':categoryA, 'y':data}).fit()
    return sm.stats.anova_lm(model)



# Chapter 7

def plot_2d(x, y, figsize=default_figsize, dpi=default_dpi):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, y)
    return None

def plot_3d(x, y, z, figsize=default_figsize, dpi=default_dpi, cmap='gist_rainbow'):
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=figsize, dpi=dpi)
    canvas = plt.subplot(111, projection='3d')
    canvas.plot_surface(X, Y, z, cmap=cmap)
    return None

def interp_1d(x, y, mode='cubic', x_tight=None, res=1000, 
              figsize=default_figsize, dpi=default_dpi):
    from scipy.interpolate import interp1d
    if x_tight is None: x_tight = np.linspace(x.min(), x.max(), res)
    func = interp1d(x, y, mode)
    y_tight = func(x_tight)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x, y)
    plt.plot(x_tight, y_tight)
    return func

def interp_2d(x, y, z, mode='cubic', x_tight=None, y_tight=None, res=1000, 
              figsize=default_figsize, dpi=default_dpi, cmap='gist_rainbow'):
    from scipy.interpolate import interp2d
    if x_tight is None: x_tight = np.linspace(x.min(), x.max(), res)
    if y_tight is None: y_tight = np.linspace(y.min(), y.max(), res)
    X_tight, Y_tight = np.meshgrid(x_tight, y_tight)
    func = intp.interp2d(x, y, z, mode)
    z_tight = func(x_tight, y_tight)
    plt.figure(figsize=figsize, dpi=dpi)
    canvas = plt.subplot(111, projection='3d')
    canvas.plot_surface(X_tight, Y_tight, z_tight, cmap=cmap)
    return func

def curvefit_1d(x, y, src_func, param_bounds=None, x_tight=None, res=1000, 
                figsize=default_figsize, dpi=default_dpi):
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    if x_tight is None: x_tight = np.linspace(x.min(), x.max(), res)
    if isinstance(src_func, int):
        params, covmat = np.polyfit(x, y, src_func), None
        func = lambda x: np.polyval(params, x)
        y_tight = func(x_tight)
    else:
        if param_bounds is not None:
            params, covmat = curve_fit(src_func, x, y, bounds=param_bounds)
        else: params, covmat = curve_fit(src_func, x, y)
        func = lambda x: src_func(x, *params)
        y_tight = func(x_tight)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x, y)
    plt.plot(x_tight, y_tight)
    return {'func': func, 'params': params, 'covmat': covmat, 
            'score': r2_score(func(x), y), 'pred': func(x)}

def curvefit_2d(x, y, z, src_func, param_bounds=None, x_tight=None, y_tight=None, res=1000, 
                figsize=default_figsize, dpi=default_dpi, cmap='gist_rainbow'):
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    xy = np.vstack((x, y))
    params, covmat = curve_fit(src_func, xy, z, bounds=bounds)
    func = lambda x: src_func(x, *params)
    if x_tight is None: x_tight = np.linspace(x.min(), x.max(), res)
    if y_tight is None: y_tight = np.linspace(y.min(), y.max(), res)
    X, Y = np.meshgrid(x_tight, y_tight)
    grid = np.array(tuple(zip(X.flatten(), Y.flatten()))).T
    z_tight = func(grid).reshape(res, res)
    X, Y = np.meshgrid(x_tight, y_tight)
    plt.figure(figsize=figsize, dpi=dpi)
    canvas = plt.subplot(111, projection='3d')
    canvas.plot_surface(X, Y, z_tight, cmap=cmap)
    return {'func': func, 'params': params, 'covmat': covmat}



# Chapter 8

def odeivp_1d(function, initial_values, t_tight=None, t_max=50, res=1000, 
              figsize=default_figsize, dpi=default_dpi):
    from scipy.integrate import odeint
    if t_tight is None: t_tight = np.linspace(0, t_max, res)
    sol_x, sol_y = odeint(function, initial_values, t_tight).T
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(t_tight, sol_x)
    plt.plot(t_tight, sol_y)
    return (sol_x, sol_y)

def odeivp_2d(function, initial_values, t_tight=None, t_max=50, res=1000, 
              figsize=default_figsize, dpi=default_dpi, cmap='gist_rainbow'):
    from scipy.integrate import odeint
    if t_tight is None: t_tight = np.linspace(0, t_max, res)
    sol_x, sol_y, sol_z = odeint(function, initial_values, t_tight).T
    plt.figure(figsize=figsize, dpi=dpi)
    canvas = plt.subplot(111, projection='3d')
    canvas.plot(sol_x, sol_y, sol_z)
    return (sol_x, sol_y, sol_z)



# Chapter 9

def normalize_standard(dataframe, minimize=[]):
    if isinstance(dataframe, pd.Series): dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe.apply(lambda col: (col-col.mean())/col.std(), axis=0)
    for criterion in minimize:
        dataframe[criterion] = 0 - dataframe[criterion]
    return dataframe

def normalize_scaletrans(dataframe, minimize=[]):
    if isinstance(dataframe, pd.Series): dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe.apply(lambda col: col/col.max(), axis=0)
    for criterion in minimize:
        dataframe[criterion] = dataframe[criterion].min() / dataframe[criterion]
    return dataframe

def normalize_vectornorm(dataframe, minimize=[]):
    if isinstance(dataframe, pd.Series): dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe.apply(lambda col: col/np.linalg.norm(col), axis=0)
    for criterion in minimize:
        dataframe[criterion] = 1 - dataframe[criterion]
    return dataframe

def cem_topsis(dataframe):
    criteria = dataframe.shape[1]
    dataframe['tmp_DistToBest'] = dataframe.apply(lambda row: np.linalg.norm((row-dataframe.max(axis=0))[:criteria]), axis=1)
    dataframe['tmp_DistToWorst'] = dataframe.apply(lambda row: np.linalg.norm((row-dataframe.min(axis=0))[:criteria]), axis=1)
    dataframe['SCORE'] = dataframe.tmp_DistToWorst/(dataframe.tmp_DistToBest+dataframe.tmp_DistToWorst)
    dataframe = dataframe.drop(columns=['tmp_DistToBest', 'tmp_DistToWorst'])
    return dataframe.sort_values(by='SCORE', ascending=False)

def cem_greycorr(dataframe, rho=0.5):
    dataframe = dataframe.max(axis=0) - dataframe
    dataframe['SCORE'] = (1-dataframe.values/(dataframe.values+rho*dataframe.values.max())).mean(axis=1)
    return dataframe.sort_values(by='SCORE', ascending=False)

def cem_entropy(dataframe):
    dataframe = dataframe.apply(lambda col: col/col.sum(), axis=0).T
    dataframe['tmp_Entropy'] = dataframe.apply(lambda row: -(row*np.log(row)).sum()/np.log(len(row)), axis=1)
    dataframe['tmp_DiffCoeff'] = 1 - dataframe.tmp_Entropy
    dataframe['tmp_WeightCoeff'] = dataframe.tmp_DiffCoeff / dataframe.tmp_DiffCoeff.sum()
    dataframe = dataframe.T
    dataframe['SCORE'] = dataframe.apply(lambda row: (row*dataframe.loc['tmp_WeightCoeff']).sum(), axis=1)
    dataframe = dataframe.drop(index=['tmp_Entropy', 'tmp_DiffCoeff', 'tmp_WeightCoeff'])
    return dataframe.sort_values(by='SCORE', ascending=False)

def cem_rsr(dataframe, weights=None):
    item, criteria = dataframe.shape
    dataframe = dataframe.rank()
    if weights is None: weights = np.ones(criteria)/criteria
    dataframe['SCORE'] = dataframe.apply(lambda row: (row*weights).sum()/item, axis=1)
    return dataframe.sort_values(by='SCORE', ascending=False)

def cem(dataframe, minimize=[]):
    scores = pd.DataFrame()
    scores['TN'] = normalize_scaletrans(cem_topsis(normalize_standard(dataframe, minimize=minimize)).SCORE)
    scores['TS'] = normalize_scaletrans(cem_topsis(normalize_scaletrans(dataframe, minimize=minimize)).SCORE)
    scores['TV'] = normalize_scaletrans(cem_topsis(normalize_vectornorm(dataframe, minimize=minimize)).SCORE)
    scores['GN'] = normalize_scaletrans(cem_greycorr(normalize_standard(dataframe, minimize=minimize)).SCORE)
    scores['GS'] = normalize_scaletrans(cem_greycorr(normalize_scaletrans(dataframe, minimize=minimize)).SCORE)
    scores['GV'] = normalize_scaletrans(cem_greycorr(normalize_vectornorm(dataframe, minimize=minimize)).SCORE)
    scores['ES'] = normalize_scaletrans(cem_entropy(normalize_scaletrans(dataframe, minimize=minimize)).SCORE)*1.5
    scores['EV'] = normalize_scaletrans(cem_entropy(normalize_vectornorm(dataframe, minimize=minimize)).SCORE)*1.5
    scores['RN'] = normalize_scaletrans(cem_rsr(normalize_standard(dataframe, minimize=minimize)).SCORE)
    scores['RS'] = normalize_scaletrans(cem_rsr(normalize_scaletrans(dataframe, minimize=minimize)).SCORE)
    scores['RV'] = normalize_scaletrans(cem_rsr(normalize_vectornorm(dataframe, minimize=minimize)).SCORE)
    scores['MEAN'] = scores.apply(lambda row: row.sum()/12, axis=1)
    return scores.sort_values(by='MEAN', ascending=False)



# Chapter 10

def graph_create(source, graph_type='Graph', draw=True, layout='shell', 
                 figsize=default_figsize, dpi=default_dpi):
    if isinstance(source, dict):
        generator = source
        source = []
        for src, dsts in generator.items():
            if isinstance(dsts[0], tuple): source.extend([(src,)+dst for dst in dsts])
            else: source.extend([(src, dst) for dst in dsts])
    try:
        graph = eval('nx.'+graph_type)(source)
    except nx.NetworkXError:
        graph = eval('nx.'+graph_type)()
        if len(source[0]) == 2: graph.add_edges_from(source)
        else: graph.add_weighted_edges_from(source)
    if draw == True: graph_draw(graph, layout=layout, figsize=figsize, dpi=dpi)
    return graph

def graph_draw(graph, layout='shell', figsize=default_figsize, dpi=default_dpi):
    pos = eval('nx.'+layout+'_layout')(graph)
    weight = nx.get_edge_attributes(graph, 'weight')
    plt.figure(figsize=figsize, dpi=dpi)
    nx.draw_networkx(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight)
    return None

def graph_sp(graph, src, dst):
    if str(type(graph))[8:16] != 'networkx': graph = graph_create(graph, draw=False)
    return nx.shortest_path(graph, src, dst, weight='weight'), \
           nx.shortest_path_length(graph, src, dst, weight='weight')

def graph_spmat(graph):
    if str(type(graph))[8:16] != 'networkx': graph = graph_create(graph, draw=False)
    splen = dict(nx.shortest_path_length(graph, weight='weight'))
    spmat = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for src, dsts in splen.items():
        for dst, dist in dsts.items():
            spmat[src-1, dst-1] = dist
    return spmat

def graph_mst(graph):
    if str(type(graph))[8:16] != 'networkx': graph = graph_create(graph, draw=False)
    mst = nx.minimum_spanning_tree(graph)
    graph_draw(mst)
    return mst

def graph_match(graph, mode='max_weight'):
    from networkx.algorithms import matching
    if str(type(graph))[8:16] != 'networkx':
        size = graph.shape[0]
        if size != graph.shape[1]:
            print("Invalid metric.")
            return None
        else:
            frame = np.zeros((size*2, size*2))
            frame[:size, size:] = graph
            graph = graph_create(frame, draw=False)
    matching = eval('matching.'+mode+'_matching')(graph)
    result = []
    total_weight = 0
    for src, dst in matching:
        total_weight += graph.get_edge_data(src, dst)['weight']
        if src > dst: src, dst = dst, src
        src += 1
        dst -= len(matching) - 1
        result.append((src, dst))
    result.sort()
    return result, total_weight

def network_create(source, draw=True, layout='shell', figsize=default_figsize, dpi=default_dpi):
    if isinstance(source, dict):
        generator = source
        source = []
        for src, dsts in generator.items():
            if isinstance(dsts[0], tuple): source.extend([(src,)+dst for dst in dsts])
            else: source.extend([(src, dst) for dst in dsts])
    if len(source[0]) == 3: 
        cpcs = source
        wgts = []
    elif len(source[0]) == 4:
        cpcs = [(src, dst, cpc) for src, dst, cpc, _ in source]
        wgts = [(src, dst, wgt) for src, dst, _, wgt in source]
    else: 
        print("Invalid metric.")
        return None
    network = nx.DiGraph()
    network.add_weighted_edges_from(cpcs, weight='capacity')
    network.add_weighted_edges_from(wgts, weight='weight')
    if draw == True: graph_draw(network, layout=layout, figsize=figsize, dpi=dpi)
    return network

def graph_pagerank(graph, redirect_prob=0.85, figsize=default_figsize, dpi=default_dpi):
    from scipy.sparse.linalg import eigs
    link = nx.to_numpy_matrix(graph)
    probs = link / np.tile(link.sum(axis=1), (1, link.shape[1]))
    probs = redirect_prob * probs + (1-redirect_prob) / link.shape[0]
    eig = eigs(probs.T, 1)[1].real.flatten()
    eig /= eig.sum()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.bar(range(1, link.shape[0]+1), eig)
    return eig



# Chapter 11

def classify_knc(train, label, inquiry):
    from sklearn.neighbors import KNeighborsClassifier as KNC
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski3', 
               'minkowski4', 'minkowski5', 'mahalanobis', 'seuclidean']
    result = {}
    for metric in metrics:
        if metric[:9] == 'minkowski': metric_real, params = metric[:9], ', p=%s' % metric[9:]
        elif metric == 'mahalanobis': metric_real, params = metric, ", metric_params={'V':np.cov(train.T)}"
        elif metric == 'seuclidean': 
            metric_real, params = metric[1:], ''
            train = (train-train.mean(axis=0)) / train.std(axis=0)
        else: metric_real, params = metric, ""
        knc_best, k_best, acc_best, k = None, 2, 0, 2
        while True:
            try:
                knc = eval("KNC(k, metric='%s'%s)" % (metric_real, params))
                knc.fit(train, label)
                acc = knc.score(train, label)
                if acc > acc_best: knc_best, k_best, acc_best = knc, k, acc
                k += 1
            except:
                break
        knc_best.fit(train, label)
        result['%s(k=%d)' % (metric, k_best)] = {'pred': knc_best.predict(inquiry), 'acc': acc_best, 
                                                 'cv': classify_crossval(knc_best, train, label)}
    return result

def classify_fisher(train, label, inquiry):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA()
    lda.fit(train, label)
    return {'fisher': {'pred': lda.predict(inquiry), 'acc': lda.score(train, label), 
                       'cv': classify_crossval(lda, train, label)}}

def classify_gaussiannb(train, label, inquiry):
    from sklearn.naive_bayes import GaussianNB as GNB
    gnb = GNB()
    gnb.fit(train, label)
    return {'gaussiannb': {'pred': gnb.predict(inquiry), 'acc': gnb.score(train, label), 
                           'cv': classify_crossval(gnb, train, label)}}

def classify_crossval(model, train, label):
    from sklearn.model_selection import cross_val_score
    result = []
    for cv in range(2, int(np.floor(np.sqrt(20)))+1):
        result.append(np.round(cross_val_score(model, train, label, cv=cv).mean(), 4))
    return result

def classify(train, label, inquiry):
    result_knc = classify_knc(train, label, inquiry)
    result_fisher = classify_fisher(train, label, inquiry)
    result_gaussiannb = classify_gaussiannb(train, label, inquiry)
    return dict(result_knc, **result_fisher, **result_gaussiannb)

def multivar_pca(data, preserve_rate=0.9, figsize=default_figsize, dpi=default_dpi):
    from sklearn.decomposition import PCA
    if isinstance(data, np.ndarray): data = pd.DataFrame(data)
    pca = PCA().fit(data.values)
    pcacmp, pcapct = pca.components_, pca.explained_variance_ratio_
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(np.arange(len(pcapct))+1, np.cumsum(pcapct))
    threshold = np.where(np.cumsum(pcapct) > preserve_rate)[0][0] + 1
    pcaval = data @ pcacmp[:threshold].T
    pcaval.index = data.index
    pcaval.columns = ['Element_%d' % (i+1) for i in np.arange(threshold)]
    pcaval['PCA_Score'] = np.dot(pcapct[:threshold]/pcapct[:threshold].sum(), pcaval.T)
    if all(pcaval.PCA_Score < 0): pcaval = -pcaval
    return {'pcaval': pd.concat((data, pcaval), axis=1).sort_values('PCA_Score', ascending=False), 
            'pcacmp': pcacmp, 'pcapct': pcapct}

def multivar_fa(data, preserve_rate=0.9, figsize=default_figsize, dpi=default_dpi):
    from sklearn.decomposition import PCA, FactorAnalysis
    if isinstance(data, pd.DataFrame):
        columns, index = data.columns, data.index
        data = data.values
    else: columns, index = None, None
    data = (data-data.mean(axis=0))/data.std(axis=0)
    data = pd.DataFrame(data, columns=columns, index=index)
    pca = PCA().fit(data)
    pcacmp, pcapct = pca.components_, pca.explained_variance_ratio_
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(np.arange(len(pcapct))+1, np.cumsum(pcapct))
    threshold = np.where(np.cumsum(pcapct) > preserve_rate)[0][0] + 1
    fa = FactorAnalysis(n_components=threshold).fit(data)
    facmp, fanos = fa.components_, fa.noise_variance_
    faval = pd.DataFrame(fa.fit_transform(data))
    faval.index = data.index
    faval.columns = ['Element_%d' % (i+1) for i in np.arange(threshold)]
    faval['FA_Score'] = np.dot(pcapct[:threshold]/pcapct[:threshold].sum(), faval.T)
    return {'faval': pd.concat((data, faval), axis=1).sort_values('FA_Score', ascending=False), 
            'pcacmp': pcacmp, 'pcapct': pcapct, 'facmp': facmp, 'fanos': fanos}

def cluster_hierarchy(data, scale='minmax_scale', metric='euclidean', method='single', 
                      figsize=default_figsize, dpi=default_dpi):
    from sklearn import preprocessing as pp
    import scipy.cluster.hierarchy as sch
    if scale is not None: data = eval('pp.'+scale)(data)
    dist = sch.distance.squareform(sch.distance.pdist(data, metric=metric))
    cluster = sch.linkage(dist, metric=metric, method=method)
    plt.figure(figsize=figsize, dpi=dpi)
    sch.dendrogram(cluster)
    return cluster

def cluster_kmeans(data, n_clusters):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    kmeans = KMeans(n_clusters=3).fit(data)
    labels, centers, SSE = kmeans.labels_, kmeans.cluster_centers_, 0
    for label in set(labels): SSE += np.sum((data[labels == label] - centers[label]) ** 2)
    silhouette = silhouette_score(data, labels)
    return {'labels': labels, 'centers': centers, 'SSE': SSE, 'silhouette_score': silhouette}

def cluster_kmeans_sse(data, max_k=12, figsize=default_figsize, dpi=default_dpi):
    from sklearn.cluster import KMeans
    TSSE = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k).fit(data)
        labels, centers, SSE = kmeans.labels_, kmeans.cluster_centers_, 0
        for label in set(labels): SSE += np.sum((data[labels == label] - centers[label]) ** 2)
        TSSE.append(SSE)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(range(2, max_k+1), TSSE)
    return {'sse_scores': TSSE}

def cluster_kmeans_silhouette(data, max_k=12, metric='euclidean', 
                              figsize=default_figsize, dpi=default_dpi):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    TSIL, best_k, best_score = [], 0, 0
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels, metric=metric)
        if score > best_score: best_k, best_score = k, score
        TSIL.append(score)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(range(2, max_k+1), TSIL)
    return {'silhouette_scores': TSIL, 'best_k': best_k, 'best_score': best_score}



# Chapter 12

def regression_linear(x, y):
    from sklearn.linear_model import LinearRegression
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    linreg = LinearRegression().fit(x, y)
    y_pred = linreg.predict(x)
    coeff, score = np.concatenate((linreg.coef_, np.array([linreg.intercept_]))), linreg.score(x, y)
    return {'model': linreg, 'coeff': coeff, 'score': score, 'y_pred': y_pred}

def regression_ridge(x, y, alphas, figsize=default_figsize, dpi=default_dpi):
    from sklearn.linear_model import Ridge, RidgeCV
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    ridgecv = RidgeCV(alphas=alphas).fit(x, y)
    ridge = Ridge(ridgecv.alpha_).fit(x, y)
    y_pred = ridge.predict(x)
    coeff, score = np.concatenate((ridge.coef_, np.array([ridge.intercept_]))), ridge.score(x, y)
    return {'model': ridge, 'alpha': ridgecv.alpha_, 'coeff': coeff, 'score': score, 'y_pred': y_pred}

def regression_ridge_manual(x, y, alphas, figsize=default_figsize, dpi=default_dpi):
    from sklearn.linear_model import Ridge
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    coeffs, scores, best_alpha, best_ridge, best_coeff, best_score = [], [], 0, None, None, 0
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(x, y)
        coeff, score = np.concatenate((ridge.coef_, np.array([ridge.intercept_]))), ridge.score(x, y)
        coeffs.append(coeff)
        scores.append(score)
        if score > best_score: 
            best_alpha, best_ridge, best_coeff, best_score = alpha, ridge, coeff, score
    coeffs = np.array(coeffs).T
    figsize_x, figsize_y = figsize
    plt.figure(figsize=(figsize_x*2, figsize_y), dpi=default_dpi)
    for dim in range(coeffs.shape[0]-1): plt.subplot(121).plot(alphas, coeffs[dim])
    plt.legend(['x%d' % (dim+1) for dim in range(coeffs.shape[0])])
    plt.subplot(122).plot(alphas, scores)
    y_pred = best_ridge.predict(x)
    return {'model': best_ridge, 'alpha': best_alpha, 'coeff': best_coeff, 'score': best_score, 'y_pred': y_pred}

def regression_lasso(x, y, alphas, figsize=default_figsize, dpi=default_dpi):
    from sklearn.linear_model import Lasso, LassoCV
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    lassocv = LassoCV(alphas=alphas).fit(x, y)
    lasso = Lasso(lassocv.alpha_).fit(x, y)
    y_pred = lasso.predict(x)
    coeff, score = np.concatenate((lasso.coef_, np.array([lasso.intercept_]))), lasso.score(x, y)
    return {'model': lasso, 'alpha': lassocv.alpha_, 'coeff': coeff, 'score': score, 'y_pred': y_pred}

def regression_lasso_manual(x, y, alphas, figsize=default_figsize, dpi=default_dpi):
    from sklearn.linear_model import Lasso
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    coeffs, scores, best_alpha, best_lasso, best_coeff, best_score = [], [], 0, None, None, 0
    for alpha in alphas:
        lasso = Lasso(alpha=alpha).fit(x, y)
        coeff, score = np.concatenate((lasso.coef_, np.array([lasso.intercept_]))), lasso.score(x, y)
        coeffs.append(coeff)
        scores.append(score)
        if score > best_score: 
            best_alpha, best_lasso, best_coeff, best_score = alpha, lasso, coeff, score
    coeffs = np.array(coeffs).T
    figsize_x, figsize_y = figsize
    plt.figure(figsize=(figsize_x*2, figsize_y), dpi=default_dpi)
    for dim in range(coeffs.shape[0]-1): plt.subplot(121).plot(alphas, coeffs[dim])
    plt.legend(['x%d' % (dim+1) for dim in range(coeffs.shape[0])])
    plt.subplot(122).plot(alphas, scores)
    y_pred = best_lasso.predict(x)
    return {'model': best_lasso, 'alpha': best_alpha, 'coeff': best_coeff, 'score': best_score, 'y_pred': y_pred}

def regression_logistic(x, y):
    from sklearn.linear_model import LogisticRegression
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    logreg = LogisticRegression().fit(x, y)
    y_pred = logreg.predict(x)
    coeff, score = np.concatenate((logreg.coef_.flatten(), logreg.intercept_)), logreg.score(x, y)
    return {'model': logreg, 'coeff': coeff, 'score': score, 'y_pred': y_pred}

def regression_logistic_continuous(x, y):
    from sklearn.metrics import r2_score
    if isinstance(x, pd.Series): x = x.values
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 2: y = y.flatten()
    y_log = np.log(y / (1 - y))
    linreg = regression_linear(x, y_log)
    y_pred_log, model, coeff, score_log = linreg.values()
    y_pred = 1 - 1 / (1 + np.exp(y_pred_log))
    score = r2_score(y, y_pred)
    return {'model': model, 'coeff': coeff, 'score_log': score_log, 'score': score, 
            'y_pred_log': y_pred_log, 'y_pred': y_pred, }



# Chapter 14

def fuzzy_composition(rel_a, rel_b, comp_mode='mult-sum'):
    if rel_a.ndim > 2 or rel_b.ndim > 2:
        print("Array dimensions too high.")
        return None
    if rel_a.ndim == 1: rel_a = rel_a.reshape(1, -1)
    if rel_b.ndim == 1: rel_b = rel_b.reshape(-1, 1)
    if rel_a.shape[1] != rel_b.shape[0]:
        print("Unmatched dimensions.")
        return None
    if   comp_mode == 'min-max':  return np.array([[np.minimum(rel_a[i], rel_b[:,j].T).max() 
                      for j in range(rel_b.shape[1])] for i in range(rel_a.shape[0])])
    elif comp_mode == 'mult-max': return np.array([[(rel_a[i] * rel_b[:,j].T).max() 
                      for j in range(rel_b.shape[1])] for i in range(rel_a.shape[0])])
    elif comp_mode == 'min-sum':  return np.array([[np.minimum(rel_a[i], rel_b[:,j].T).sum() 
                      for j in range(rel_b.shape[1])] for i in range(rel_a.shape[0])])
    elif comp_mode == 'mult-sum': return np.array([[np.dot(rel_a[i], rel_b[:,j].T) 
                      for j in range(rel_b.shape[1])] for i in range(rel_a.shape[0])])
    else:
        print("Illegal composition mode.")
        return None

def fuzzy_categorize(benchmarks, inquiry, dist_ord=2):
    n, scores = len(inquiry), []
    for benchmark in benchmarks:
        scores.append(1-np.linalg.norm(benchmark-inquiry, ord=dist_ord)/n**(1/dist_ord))
    scores = np.array(scores)
    return {'scores': scores, 'category': scores.argmax(), 'highest_score': scores.max()}

def fuzzy_tcm(mat):
    tcm = fuzzy_composition(mat, mat, comp_mode='min-max')
    while (tcm != mat).any():
        mat = tcm
        tcm = fuzzy_composition(mat, mat, comp_mode='min-max')
    return tcm

def fuzzy_cluster_hierarchy(mat, metric='euclidean', method='single', 
                      figsize=default_figsize, dpi=default_dpi):
    from sklearn import preprocessing as pp
    import scipy.cluster.hierarchy as sch
    tcm = np.triu(1-fuzzy_tcm(mat), 1)
    tcm = tcm[tcm != 0]
    cluster = sch.linkage(tcm, metric=metric, method=method)
    plt.figure(figsize=figsize, dpi=dpi)
    sch.dendrogram(cluster)
    return cluster

def fuzzy_cluster_cmeans(samples, num_clusters, expord=2, tolerance=10e-18, maxiter=1000):
    from skfuzzy.cluster import cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(samples.T, c=num_clusters, m=expord, 
                                        error=tolerance, maxiter=maxiter, seed=0)
    return {'centers': cntr, 'labels': u.argmax(axis=0)+1, 'confidence': u.T, 
            'distance': d, 'obj_func_history': jm, 'iter_num': p, 'partition_coeff': fpc}



# Chapter 15

def graymodel_curvefit_1d(y, param_bounds=None, x_tight=None, res=1000, 
                figsize=default_figsize, dpi=default_dpi):
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    x = np.arange(len(y)) + 1
    src_func = lambda t, a, b, c: a*np.exp(b*t)+c
    func, params, covmat, pred_acc, score_acc = curvefit_1d(x, np.cumsum(y), src_func, 
        param_bounds=param_bounds, x_tight=x_tight, res=res, figsize=figsize, dpi=dpi).values()
    pred = np.hstack([pred_acc[0], np.diff(pred_acc)])
    return {'func': func, 'params': params, 'covmat': covmat, 
            'score_acc': score_acc, 'score': r2_score(pred, y), 'pred': pred}

def graymodel_pred_1_1(y, figsize=default_figsize, dpi=default_dpi):
    from sklearn.metrics import r2_score
    y_acc = np.cumsum(y)
    z = (y_acc[:-1] + y_acc[1:]) / 2
    stepwise_ratio = y[:-1] / y[1:]
    bnd_act_l, bnd_act_h = (stepwise_ratio.min(), stepwise_ratio.max())
    bnd_req_l, bnd_req_h = (np.exp(-2/(len(y)+1)), np.exp(2/(len(y)+1)))
    if bnd_act_l < bnd_req_l or bnd_act_h > bnd_req_h: print("OVERFLOW WARNING: Bounds exceeded")
    mat = np.c_[-z, np.ones(len(y)-1)]
    a, b = np.linalg.pinv(mat) @ y[1:]
    t, x = sp.Symbol('t'), sp.Function('x')
    eqn = x(t).diff(t)+a*x(t)-b
    func_acc = sp.lambdify(t, sp.dsolve(eqn, ics={x(0): y[0]}).args[1], 'numpy')
    func = lambda x: func_acc(x) - func_acc(x-1)
    y_pred_acc = func_acc(np.arange(len(y)+1))
    y_pred = np.hstack([y_pred_acc[0], np.diff(y_pred_acc)])
    figsize_x, figsize_y = figsize
    plt.figure(figsize=(figsize_x*2, figsize_y), dpi=default_dpi)
    plt.subplot(121).plot(y_acc)
    plt.subplot(121).plot(y_pred_acc[:-1])
    plt.subplot(122).plot(y)
    plt.subplot(122).plot(y_pred[:-1])
    return {'func': func, 'eqn_params': np.array((a, b)), 
            'score_acc': r2_score(y_acc, y_pred_acc[:-1]), 'score': r2_score(y, y_pred[:-1]), 'y_pred': y_pred, 
            'bound_actual': np.array((bnd_act_l, bnd_act_h)), 'bound_required': np.array((bnd_req_l, bnd_req_h))}

def graymodel_pred_1_n(y, reliance, figsize_x=12, dpi=default_dpi):
    from scipy.integrate import odeint
    from sklearn.metrics import r2_score
    y_acc = np.cumsum(y, axis=0)
    z = (y_acc[:-1] + y_acc[1:]) / 2
    stepwise_ratio = y[:-1] / y[1:]
    bnd_act = np.c_[stepwise_ratio.min(axis=0), stepwise_ratio.max(axis=0)]
    bnd_req = np.array((np.exp(-2/(len(y)+1)), np.exp(2/(len(y)+1))))
    warning = np.arange(y.shape[1])[np.logical_or(bnd_act.T[0] < bnd_req[0], bnd_act.T[1] > bnd_req[1])]
    if len(warning): print("OVERFLOW WARNING: Bounds exceeded for criteria %s" % warning)
    eqn_params, score_acc, score = [], [], []
    for var in range(y.shape[1]):
        if isinstance(reliance[var], int):
            mat = np.c_[z[:,var], np.ones(len(y)-1)]
        else:
            mat = np.c_[z[:,reliance[var][0]]]
            for factor in reliance[var][1:]:
                mat = np.c_[mat, z[:,factor]]
        eqn_params.append(np.linalg.pinv(mat) @ y[1:,var])
    def func(x, t):
        ret_vals = []
        for var, param in enumerate(eqn_params):
            if isinstance(reliance[var], int):
                ret_val = param[0]*x[var]+param[1]
            else:
                ret_val = 0
                for i, factor in enumerate(reliance[var]):
                    ret_val += param[i]*x[factor]
            ret_vals.append(ret_val)
        return np.array(ret_vals)
    x = np.arange(len(y)+1)
    y_pred_acc = odeint(func, y[0], x)
    y_pred = np.vstack([y[0], np.diff(y_pred_acc, axis=0)])
    plt.figure(figsize=(figsize_x, figsize_x/y.shape[1]*2), dpi=dpi)
    for var in range(y.shape[1]):
        plt.subplot(200+y.shape[1]*10+var+1).plot(y_acc[:,var])
        plt.subplot(200+y.shape[1]*10+var+1).plot(y_pred_acc[:-1,var])
        plt.subplot(200+y.shape[1]*11+var+1).plot(y[:,var])
        plt.subplot(200+y.shape[1]*11+var+1).plot(y_pred[:-1,var])
        score_acc.append(r2_score(y_acc[:,var], y_pred_acc[:-1,var]))
        score.append(r2_score(y[:,var], y_pred[:-1,var]))
    return {'func': func, 'eqn_params': eqn_params, 
            'score_acc': score_acc, 'score': score, 'y_pred': y_pred, 
            'bound_actual': bnd_act, 'bound_required': bnd_req}

def graymodel_pred_2_1(y, figsize=default_figsize, dpi=default_dpi):
    from sklearn.metrics import r2_score
    y_acc = np.cumsum(y)
    z = (y_acc[:-1] + y_acc[1:]) / 2
    stepwise_ratio = y[:-1] / y[1:]
    bnd_act_l, bnd_act_h = (stepwise_ratio.min(), stepwise_ratio.max())
    bnd_req_l, bnd_req_h = (np.exp(-2/(len(y)+1)), np.exp(2/(len(y)+1)))
    if bnd_act_l < bnd_req_l or bnd_act_h > bnd_req_h: print("OVERFLOW WARNING: Bounds exceeded")
    mat = np.c_[-y[1:], -z, np.ones(len(y)-1)]
    a, b, c = np.linalg.pinv(mat) @ np.diff(y)
    t, x = sp.Symbol('t'), sp.Function('x')
    eqn = x(t).diff(t,2)+a*x(t).diff(t)+b*x(t)-c
    func_acc = sp.lambdify(t, sp.dsolve(eqn, ics={x(0): y[0], x(len(y)-1): y_acc[-1]}).args[1], 'numpy')
    func = lambda x: func_acc(x) - func_acc(x-1)
    y_pred_acc = func_acc(np.arange(len(y)+1))
    y_pred = np.hstack([y_pred_acc[0], np.diff(y_pred_acc)])
    figsize_x, figsize_y = figsize
    plt.figure(figsize=(figsize_x*2, figsize_y), dpi=default_dpi)
    plt.subplot(121).plot(y_acc)
    plt.subplot(121).plot(y_pred_acc[:-1])
    plt.subplot(122).plot(y)
    plt.subplot(122).plot(y_pred[:-1])
    return {'func': func, 'eqn_params': np.array((a, b)), 
            'score_acc': r2_score(y_acc, y_pred_acc[:-1]), 'score': r2_score(y, y_pred[:-1]), 'y_pred': y_pred, 
            'bound_actual': np.array((bnd_act_l, bnd_act_h)), 'bound_required': np.array((bnd_req_l, bnd_req_h))}



# Chapter 17

def coord_to_dist(coords, dist_mode='globe'):
    N = coords.shape[0]
    dists = np.zeros([N, N])
    if dist_mode == 'globe':
        coords = np.radians(coords)
        for src in range(N):
            for dst in range(N):
                (x1, y1), (x2, y2) = coords[src], coords[dst]
                dists[src, dst] = 6370 * np.lib.scimath.arccos(np.cos(x1-x2)*np.cos(y1)*np.cos(y2)
                                                               +np.sin(y1)*np.sin(y2)).real
        return dists
    elif dist_mode == 'plain':
        for src in range(N):
            for dst in range(N):
                (x1, y1), (x2, y2) = coords[src], coords[dst]
                dists[src, dst] = np.sqrt((x1-x2)**2+(y1-y2)**2)
        return dists
    else:
        print("Invalid distance mode.")
        return None

def intel_simanneal_optimizer(coords, temp=1, alpha=0.999, tol=10e-50, max_iter=100000, guess_num=10000, 
                              dist_mode='globe', figsize=default_figsize, dpi=default_dpi):
    N, best_path, best_length = coords.shape[0], [], np.inf
    dists = coord_to_dist(coords, dist_mode=dist_mode)
    for seed in range(guess_num):
        np.random.seed(seed)
        path = np.arange(1, N-1)
        np.random.shuffle(path)
        path = np.r_[0, path, N-1]
        length = 0
        for step in range(N-1):
            length += dists[path[step], path[step+1]]
        if length < best_length: best_path, best_length = path, length
    path = best_path
    for seed in tqdm(range(max_iter)):
        np.random.seed(seed)
        u, v = np.random.randint(1, N-1, 2)
        if u > v: u, v = v, u
        diff = (dists[path[u-1],path[v]]+dists[path[u],path[v+1]]) - (dists[path[u-1],path[u]]+dists[path[v],path[v+1]])
        if diff < 0 or np.exp(-diff/temp) >= np.random.rand(1): path = np.r_[path[:u], path[v:u-1:-1], path[v+1:]]
        temp *= alpha
        if temp < tol: break
    length = 0
    for step in range(N-1):
        length += dists[path[step], path[step+1]]
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(coords[path,0], coords[path,1], '*-')
    return {'path': path, 'length': length}

def intel_genetic_optimizer(coords, population, generation=10, mut_prob=0.1, 
                            dist_mode='globe', figsize=default_figsize, dpi=default_dpi):
    N, paths = coords.shape[0], []
    dists = coord_to_dist(coords, dist_mode=dist_mode)
    for seed in tqdm(range(population)):
        np.random.seed(seed)
        path = np.arange(1, N-1)
        np.random.shuffle(path)
        path = np.r_[0, path, N-1]
        flag = 1
        while flag != 0:
            flag = 0
            for p in range(1, N-3):
                for q in range(p+1, N-2):
                    if dists[path[p], path[q]] + dists[path[p+1], path[q+1]] \
                     < dists[path[p], path[p+1]] + dists[path[q], path[q+1]]:
                        path[p+1:q+1] = path[q:p:-1]
                        flag = 1
        path[path] = np.arange(N)
        paths.append(path)
    paths = np.array(paths) / (N-1)
    for seed in tqdm(range(generation)):
        np.random.seed(seed)
        A = paths.copy()
        path = np.arange(population)
        np.random.shuffle(path)
        cross = np.random.randint(2, N-2, population)
        for i in range(0, population, 2):
            A[path[i], cross[i]:N-1], A[path[i+1], cross[i]:N-1] = A[path[i+1], cross[i]:N-1], A[path[i], cross[i]:N-1]
        mutation = np.where(np.random.rand(population) < 0.01)[0]
        nextgen = np.r_[paths, A, A[mutation]]
        index = np.argsort(nextgen, axis=1)
        lengths = np.zeros(nextgen.shape[0])
        for i in range(nextgen.shape[0]):
            for j in range(N-1):
                lengths[i] = lengths[i] + dists[index[i,j], index[i,j+1]]
        indexL = np.argsort(lengths)
        paths = nextgen[indexL]
    path, length = index[indexL[0]], lengths[indexL[0]]
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(coords[path,0], coords[path,1], '*-')
    return {'path': path, 'length': length}



# Chapter 18

def timeseries_moving_average(data, N, figsize=default_figsize, dpi=default_dpi):
    from sklearn.metrics import r2_score
    preds = np.convolve(np.ones(N)/N, data)[N-1:1-N]
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(data[N:])
    plt.plot(preds[:-1])
    plt.legend(['Data', 'Pred'])
    return {'score': r2_score(data[N:], preds[:-1]), 'pred': preds[-1], 'moving_average': preds[:-1]}

def timeseries_exp_smoothing(data, alpha, initial_avg=1, 
                             figsize=default_figsize, dpi=default_dpi):
    from sklearn.metrics import r2_score
    M = np.zeros_like(data)
    M[0] = data[:initial_avg].mean()
    for i in range(1, len(data)):
        M[i] = alpha * data[i] + (1-alpha) * M[i-1]
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(data)
    plt.plot(M)
    plt.legend(['Data', 'Pred'])
    return {'score': r2_score(data, M), 'pred': alpha * data[-1] + (1-alpha) * M[-1], 'smoothened': M}

def timeseries_exp_smoothing_ord2(data, alpha, initial_avg=1, num_preds=1, 
                                  figsize=default_figsize, dpi=default_dpi):
    from sklearn.metrics import r2_score
    M1 = np.zeros_like(data)
    M2 = np.zeros_like(data)
    preds = np.zeros_like(data)
    M1[0] = M2[0] = data[:initial_avg].mean()
    for i in range(1, len(data)):
        M1[i] = alpha * data[i] + (1-alpha) * M1[i-1]
        M2[i] = alpha * M1[i] + (1-alpha) * M2[i-1]
        preds[i-1] = 2*M1[i-1] - M2[i-1] + alpha/(1-alpha)*(M1[i-1]-M2[i-1])
    a, b = 2*M1[-1] - M2[-1], alpha/(1-alpha)*(M1[-1]-M2[-1])
    func = lambda x: a + b * x
    preds = np.r_[preds[:-1], a + b]
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(data)
    plt.plot(preds)
    plt.legend(['Data', 'Pred'])
    return {'score': r2_score(data, preds), 'func': func, 'preds': func(np.arange(num_preds)+1), 
            'M1': M1, 'M2': M2, 'smoothened': preds}

def timeseries_arima(data, order, figsize=(16, 24), dpi=200):
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import r2_score
    import warnings
    warnings.filterwarnings('error')
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series): data = data.values
    model = sm.tsa.ARIMA(data, order=order).fit()
    params = pd.Series(model.params, index=model.param_names)
    preds = model.predict()
    score = r2_score(data, preds)
    residual = model.resid
    ics = {'aic': model.aic, 'bic': model.bic, 'hqic': model.hqic}
    print("ARIMA(%d, %d, %d) achieves %2.6f%% prediction accuracy." 
          % (order[0], order[1], order[2], np.sqrt(score)*100))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    try: fig = plot_acf(data, ax=plt.subplot(331))
    except: pass
    try: fig = plot_pacf(data, ax=plt.subplot(332), method='yw')
    except: pass
    pd.Series(residual).plot(kind='kde', ax=fig.add_subplot(333), title='Residual KDE', legend=False)
    plt.subplot(312).plot(data)
    plt.subplot(312).plot(preds)
    plt.legend(['Data', 'Pred'])
    plt.title('Prediction')
    pd.Series(residual).plot(ax=fig.add_subplot(325), title='Residual', legend=False)
    plt.legend(['Residual'])
    unit = list(range(data.min().astype(int), data.max().astype(int)+1))
    plt.subplot(326).plot(unit, unit)
    plt.subplot(326).scatter(data, preds)
    plt.title('Projected Residual')
    return {'model': model, 'pdq': order, 'score': score, 
            'preds': preds, 'residual': residual, 'params': params, 'ics': ics}

def timeseries_arima_auto(data, max_order, figsize=(16, 24), dpi=200):
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import r2_score
    import warnings
    warnings.filterwarnings('error')
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series): data = data.values
    max_p, max_d, max_q = np.array(max_order) + 1
    best_p, best_d, best_q, best_model, best_preds, best_score = 0, 0, 0, None, None, 0
    for attempt in tqdm(range(1, max_p*max_d*max_q)):
        p, d, q = attempt//(max_d*max_q), attempt%(max_d*max_q)//max_q, attempt%max_q
        try: model = sm.tsa.ARIMA(data, order=(p, d, q)).fit()
        except: pass
        else:
            preds = model.predict()
            score = r2_score(data, preds)
            if score > best_score:
                best_p, best_d, best_q, best_model, best_preds, best_score = p, d, q, model, preds, score
    params = pd.Series(best_model.params, index=best_model.param_names)
    residual = best_model.resid
    ics = {'aic': best_model.aic, 'bic': best_model.bic, 'hqic': best_model.hqic}
    print("ARIMA(%d, %d, %d) achieves %2.6f%% prediction accuracy." 
          % (best_p, best_d, best_q, np.sqrt(best_score)*100))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    try: fig = plot_acf(data, ax=plt.subplot(331))
    except: pass
    try: fig = plot_pacf(data, ax=plt.subplot(332), method='yw')
    except: pass
    pd.Series(residual).plot(kind='kde', ax=fig.add_subplot(333), title='Residual KDE', legend=False)
    plt.subplot(312).plot(data)
    plt.subplot(312).plot(best_preds)
    plt.legend(['Data', 'Pred'])
    plt.title('Prediction')
    pd.Series(residual).plot(ax=fig.add_subplot(325), title='Residual', legend=False)
    plt.legend(['Residual'])
    unit = list(range(data.min().astype(int), data.max().astype(int)+1))
    plt.subplot(326).plot(unit, unit)
    plt.subplot(326).scatter(data, best_preds)
    plt.title('Projected Residual')
    return {'model': best_model, 'pdq': (best_p, best_d, best_q), 'score': best_score, 
            'preds': best_preds, 'residual': residual, 'params': params, 'ics': ics}



# Chapter 19

def svm_classify(train, label, C=1, kernel='rbf', gamma='scale'):
    from sklearn.svm import SVC
    classifier = SVC(C=C, kernel=kernel, random_state=0).fit(train, label)
    preds = classifier.predict(train)
    return {'classifier': classifier, 'score': classifier.score(train, label), 
            'preds': preds, 'misclassified': np.where(preds != label)[0]}

def svm_classify_auto(train, label, verbose=0, C_max=20, 
                      kernel_range=['linear', 'poly', 'rbf', 'sigmoid'], gamma_range=['scale', 'auto']):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    params = {'C': np.arange(1, C_max+1), 'kernel': kernel_range, 'gamma': gamma_range}
    classifier = GridSearchCV(SVC(random_state=0), params, verbose=verbose).fit(train, label)
    best_params = classifier.best_params_
    preds = classifier.predict(train)
    return {'classifier': SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'], random_state=0), 
            'score': classifier.score(train, label), 'preds': preds, 'misclassified': np.where(preds != label)[0]}

def svm_regress(x, y, figsize=(16, 8), dpi=200, C=1, kernel='rbf', gamma='auto'):
    from sklearn.svm import SVR
    if x.ndim == 1: x = x.reshape(-1, 1)
    regressor = SVR(C=C, kernel=kernel, gamma=gamma).fit(x, y)
    preds = regressor.predict(x)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x, y)
    plt.plot(x, preds, 'r')
    plt.legend(['Accurate', 'Predicted'])
    return {'regressor': regressor, 'score': regressor.score(x, y), 'preds': preds}

def svm_regress_auto(x, y, verbose=0, figsize=(16, 8), dpi=200, C_max=5, 
                     kernel_range=['linear', 'rbf', 'sigmoid'], gamma_range=['scale', 'auto']):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    if x.ndim == 1: x = x.reshape(-1, 1)
    params = {'C': np.arange(1, C_max+1), 'kernel': kernel_range, 'gamma': gamma_range}
    regressor = GridSearchCV(SVR(), params, verbose=verbose).fit(x, y)
    best_params = regressor.best_params_
    preds = regressor.predict(x)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x, y)
    plt.plot(x, preds, 'r')
    plt.legend(['Accurate', 'Predicted'])
    return {'regressor': SVR(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma']), 
            'score': regressor.score(x, y), 'preds': preds}

