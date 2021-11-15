from entropy_estimators import *
from tqdm import tqdm 
import numpy as np

def cmim(X, y, **kwargs):

    # TODO: 1) Implement threshold 2) remove trailing value

    """
    This function implements the CMIM feature selection.
    The scoring criteria is calculated based on the formula j_cmim=I(f;y)-max_j(I(fj;f)-I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature others aren't sorted
    J_CMIM: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMIM = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)

    # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
    # we assign an extreme small value to max[i] ito make it is smaller than possible value of max(I(fj;f)-I(fj;f|y))
    max = -10000000*np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmim = 1
    # itr_ = 0

    p_bar = tqdm(total=n_features) if not is_n_selected_features_specified else tqdm(total=n_selected_features)

    while True:
        p_bar.update(1)
        # itr_ += 1
        # print(itr_)
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMIM.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break

        else:
            if j_cmim <= 0:
                break

        # we assign an extreme small value to j_cmim to ensure it is smaller than all possible values of j_cmim
        j_cmim = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2-t3 > max[i]:
                        max[i] = t2-t3
                # calculate j_cmim for feature i (not in F)
                t = t1[i] - max[i]
                # record the largest j_cmim and the corresponding feature index
                if t > j_cmim:
                    j_cmim = t
                    idx = i
        F.append(idx)
        J_CMIM.append(j_cmim)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    p_bar.close()
    return np.array(F), np.array(J_CMIM), np.array(MIfy)

## --------------------- Test --------------------------------
if __name__ == "__main__":
    x, y = [], []
    for i in range(10):
        x.append(i*2)
        y.append(i)
    x, y = np.array([x], dtype=int), np.array(y, dtype=int)
    cmim(x, y)
## ----------------------------------------------------------------