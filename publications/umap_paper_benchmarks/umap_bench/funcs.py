import matplotlib.pyplot as plt
import numpy as np

import os

import time
import numpy as np

import pickle

from cuml.metrics import trustworthiness

TRUST_BATCH_SIZE = 5000

def maybe_get_results(results, key):
    return results[key] if key in results else {}
        
def draw_chart(model, X, y, dataset, model_name, classes=None):
    
    embedding = model.fit_transform(X, y)
    
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(embedding[:,1], embedding[:,0], s=0.3, c=y, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    cbar.set_ticks(np.arange(10))
    if classes is not None:
        cbar.set_ticklabels(classes)
    plt.title("%s Embedded via %s" % (dataset, model_name));
    
    
def _run_build_and_train_once(model_class, X, y=None, kwargs={}, knn_graph=None, verbose=False, eval_trust=True):
    
    results = {}
    extra_args = {}
    if knn_graph is not None:
        extra_args["knn_graph"] = knn_graph

    if verbose:
        print("Unsupervised")
    model = model_class(**kwargs)

    try:
        start = time.time()
        embeddings = model.fit_transform(X, **extra_args)
        end = time.time() - start

        if verbose:
            print("Time: "+ str(end))
    
        n_neighbors = model.n_neighbors
        del model

        if eval_trust:
            if verbose:
                print("Done. Evaluating trustworthiness")
            trust = trustworthiness(X, embeddings, n_neighbors=n_neighbors, batch_size=TRUST_BATCH_SIZE)
        else:
            trust = None
        
        if verbose:
            print(str(trust))
        results["unsupervised"] = {"time": end, "trust": trust}
    except:
        import traceback
        traceback.print_exc()
    
    # Supervised

    if y is not None:
        if verbose:
            print("Supervised")
        kwargs["target_metric"] = "categorical"
        model = model_class(**kwargs)

        try:
            start = time.time()
            embeddings = model.fit_transform(X, y, **extra_args)
            end = time.time() - start


            n_neighbors = model.n_neighbors
            del model

            if eval_trust:
                if verbose:
                    print("Done. Evaluating trustworthiness")
                trust = trustworthiness(X, embeddings, n_neighbors=n_neighbors, batch_size=TRUST_BATCH_SIZE)
            else:
                trust = None

            if verbose:
                print(str(trust))
                print("Time: "+ str(end))

            results["supervised"] = {"time": end, "trust": trust}
        except:
            import traceback
            traceback.print_exc()
    
    # Transform
    

    if verbose:
        print("Transform")
    model = model_class(**kwargs)

    try:
        
        if knn_graph is not None:
            model.fit(X)
            start = time.time()
            embeddings = model.transform(X, **extra_args)
            end = time.time() - start
        else:
            model.fit(X, knn_graph=knn_graph)
            start = time.time()
            embeddings = model.transform(X, knn_graph=knn_graph)
            end = time.time() - start
            

        n_neighbors = model.n_neighbors
        del model

        if eval_trust:
            if verbose:
                print("Done. Evaluating trustworthiness")
            trust = trustworthiness(X, embeddings, n_neighbors=n_neighbors, batch_size=TRUST_BATCH_SIZE)
        else:
            trust = None

        if verbose:
            print(str(trust))
            print("Time: "+ str(end))
        results["xform"] = {"time": end, "trust": trust}        
    except:
        import traceback
        traceback.print_exc()
        
    return results


def build_and_train(model_class, X, y=None, kwargs={}, n_trials=4, knn_graph=None, verbose=False, eval_trust=True):
    
    results = []
    
    for trial in range(n_trials):
        results.append(_run_build_and_train_once(model_class, X, y=y, kwargs=kwargs, 
                                                 knn_graph=knn_graph, verbose=verbose, 
                                                 eval_trust=eval_trust))
    return results

def store_results(results, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def maybe_load_results(filename):
    # Load a results file if it exists, otherwise load empty dictionary"
    return pickle.load( open(filename, "rb" ) ) if os.path.exists(filename) else {}


def perform_n_samples_test(model, X, precompute_knn=True, start_samples=1024, n_indep=10, n_trials=1, n_components=2):
    import math
    results = {}
    s = np.linspace(start_samples, X.shape[0], n_indep)
    for n_samples in s:
        print("Testing " + str(n_samples))
        samples = np.random.choice(np.arange(X.shape[0]), math.floor(n_samples))
        X_sampled = X[samples]
        
        if precopute_knn:
            from cuml.neighbors import NearestNeighbors
            import cupy as cp
            d, i = NearestNeighbors(n_neighbors=15).fit(X_sampled).kneighbors(X_sampled)
            knn_graph = cp.sparse.coo_matrix((cp.asarray(d.ravel()), (cp.repeat(cp.arange(d.shape[0]), 15), cp.asarray(i.ravel()))))
        else:
            knn_graph = None
        
        results[n_samples] = build_and_train(model, 
                                             X_sampled,
                                             y=None,
                                             kwargs={"n_components": n_components},
                                             knn_graph=knn_graph, 
                                             verbose=True, 
                                             n_trials=n_trials, 
                                             eval_trust=False)
        
    return results

def perform_n_components_test(model, X, model_name, start_components=2, stop_components=1024):

    import math

    n_components = np.linspace(start_components, stop_components, 3)

    print(n_components)

    for components in n_components:
        print("Testing " + str(math.floor(components)) + " components")
        scale_results[model_name + "_" + str(math.floor(components)) + "_components"] = \
            perform_n_samples_test(model, X, n_components=math.floor(components))
        store_results(scale_results, "results/scale_results.pickle")