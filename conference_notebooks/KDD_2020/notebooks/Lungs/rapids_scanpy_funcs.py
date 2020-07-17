#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cuml
import cupy as cp
import cudf

import numpy as np
import scipy
import math

import dask.array as da

from cuml.linear_model import LinearRegression


def scale(normalized, max_value=10):
    mean = normalized.mean(axis=0)
    stddev = cp.sqrt(normalized.var(axis=0))
    
    normalized -= mean
    normalized *= 1/stddev
    
    normalized[normalized>10] = 10
    
    return normalized


def _regress_out_chunk(X, y):
    """
    Performs a data_cunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result. 
    """
    output = []
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y, convert_dtype=True)
    return y.reshape(y.shape[0],) - lr.predict(X).reshape(y.shape[0])
    


def normalize_total(filtered_cells, target_sum):
    sums = np.array(target_sum / filtered_cells.sum(axis=1)).ravel()

    normalized = filtered_cells.multiply(sums[:, np.newaxis]) # Done on host for now
    normalized = cp.sparse.csr_matrix(normalized)
    
    return normalized


def regress_out(normalized, n_counts, percent_mito, verbose=False):
    
    regressors = cp.ones((n_counts.shape[0]*3)).reshape((n_counts.shape[0], 3), order="F")

    regressors[:, 1] = n_counts
    regressors[:, 2] = percent_mito
    
    for i in range(normalized.shape[1]):
        if verbose and i % 500 == 0:
            print("Regressed %s out of %s" %(i, normalized.shape[1]))
        X = regressors
        y = normalized[:,i]
        _regress_out_chunk(X, y)
    
    return normalized


def filter_cells(sparse_gpu_array, min_genes, max_genes, rows_per_batch=10000):
    n_batches = math.ceil(sparse_gpu_array.shape[0] / rows_per_batch)
    print("Running %d batches" % n_batches)
    filtered_list = []
    for batch in range(n_batches):
        batch_size = rows_per_batch
        start_idx = batch * batch_size
        stop_idx = min(batch * batch_size + batch_size, sparse_gpu_array.shape[0])
        arr_batch = sparse_gpu_array[start_idx:stop_idx]
        filtered_list.append(_filter_cells(arr_batch, 
                                            min_genes=min_genes, 
                                            max_genes=max_genes))

    return scipy.sparse.vstack(filtered_list)


def _filter_cells(sparse_gpu_array, min_genes, max_genes):
    degrees = cp.diff(sparse_gpu_array.indptr)
    query = ((min_genes <= degrees) & (degrees <= max_genes)).ravel()
    return sparse_gpu_array.get()[query.get()]


def filter_genes(sparse_gpu_array, genes_idx, min_cells=0):
    thr = np.asarray(sparse_gpu_array.sum(axis=0) >= min_cells).ravel()
    filtered_genes = sparse_gpu_array[:,thr]
    genes_idx = genes_idx[np.where(thr)[0]]
    
    return filtered_genes, genes_idx.reset_index(drop=True)


def select_groups(labels, groups_order_subset='all'):
    """Get subset of groups in adata.obs[key].
    """
    
    adata_obs_key = labels
    groups_order = labels.cat.categories
    groups_masks = cp.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories):
        # if the name is not found, fallback to index retrieval
        if labels.cat.categories[iname] in labels.cat.codes:
            mask = labels.cat.categories[iname] == labels.cat.codes
        else:
            mask = iname == labels.cat.codes
        groups_masks[iname] = mask.values
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                cp.where(cp.array(labels.cat.categories.to_array().astype("int32")) == int(name))[0][0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = cp.where(
                cp.in1d(
                    cp.arange(len(labels.cat.categories)).astype(str),
                    cp.array(groups_order_subset),
                )
            )[0]
            
        groups_ids = [groups_id.item() for groups_id in groups_ids]
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_array().astype(int)
    else:
        groups_order_subset = groups_order.to_array()
    return groups_order_subset, groups_masks


def rank_genes_groups(
    X,
    labels, # louvain results
    var_names,
    groupby =  str,
    groups = None,
    reference = 'rest',
    n_genes = 100,
    key_added = None,
    layer = None,
    **kwds,
):

    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"

    import time
    
    start = time.time()
    
    # for clarity, rename variable
    if groups == 'all':
        groups_order = 'all'
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != 'rest' and reference not in set(groups_order):
            groups_order += [reference]
    if (
        reference != 'rest'
        and reference not in set(labels.cat.categories)
    ):
        cats = labels.cat.categories.tolist()
        raise ValueError(
            f'reference = {reference} needs to be one of groupby = {cats}.'
        )

    groups_order, groups_masks = select_groups(labels, groups_order)
    
    original_reference = reference
    
    n_vars = len(var_names)

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    if n_genes_user > X.shape[1]:
        n_genes_user = X.shape[1]
    # in the following, n_genes is simply another name for the total number of genes
    n_genes = X.shape[1]

    n_groups = groups_masks.shape[0]
    ns = cp.zeros(n_groups, dtype=int)
    for imask, mask in enumerate(groups_masks):
        ns[imask] = cp.where(mask)[0].size
    if reference != 'rest':
        ireference = cp.where(groups_order == reference)[0][0]
    reference_indices = cp.arange(n_vars, dtype=int)

    rankings_gene_scores = []
    rankings_gene_names = []
    rankings_gene_logfoldchanges = []
    rankings_gene_pvals = []
    rankings_gene_pvals_adj = []

#     if 'log1p' in adata.uns_keys() and adata.uns['log1p']['base'] is not None:
#         expm1_func = lambda x: np.expm1(x * np.log(adata.uns['log1p']['base']))
#     else:
#         expm1_func = np.expm1

    # Perform LogReg
        
    # if reference is not set, then the groups listed will be compared to the rest
    # if reference is set, then the groups listed will be compared only to the other groups listed
    from cuml.linear_model import LogisticRegression
    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception('Cannot perform logistic regression on a single cluster.')
    grouping_mask = labels.astype('int').isin(cudf.Series(groups_order))
    grouping = labels.loc[grouping_mask]
    
    X = X[grouping_mask.values, :]  # Indexing with a series causes issues, possibly segfault
    y = labels.loc[grouping]
    
    clf = LogisticRegression(**kwds)
    clf.fit(X.get(), grouping.to_array().astype('float32'))
    scores_all = cp.array(clf.coef_).T
    
    for igroup, group in enumerate(groups_order):
        if len(groups_order) <= 2:  # binary logistic regression
            scores = scores_all[0]
        else:
            scores = scores_all[igroup]

        partition = cp.argpartition(scores, -n_genes_user)[-n_genes_user:]
        partial_indices = cp.argsort(scores[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]
        rankings_gene_scores.append(scores[global_indices].get())  ## Shouldn't need to take this off device
        rankings_gene_names.append(var_names[global_indices].to_pandas())
        if len(groups_order) <= 2:
            break

    groups_order_save = [str(g) for g in groups_order]
    if (len(groups) == 2):
        groups_order_save = [g for g in groups_order if g != reference]
    
    print("Ranking took (GPU): " + str(time.time() - start))
    
    start = time.time()
    
    scores = np.rec.fromarrays(
        [n for n in rankings_gene_scores],
        dtype=[(rn, 'float32') for rn in groups_order_save],
    )
    
    names = np.rec.fromarrays(
        [n for n in rankings_gene_names],
        dtype=[(rn, 'U50') for rn in groups_order_save],
    )
    
    print("Preparing output np.rec.fromarrays took (CPU): " + str(time.time() - start))
    print("Note: This operation will be accelerated in a future version")
    
    return scores, names, original_reference
