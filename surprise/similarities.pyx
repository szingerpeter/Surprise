"""
The :mod:`similarities <surprise.similarities>` module includes tools to
compute similarity metrics between users or items. You may need to refer to the
:ref:`notation_standards` page. See also the
:ref:`similarity_measures_configuration` section of the User Guide.

Available similarity measures:

.. autosummary::
    :nosignatures:

    cosine
    mad
    msd
    pearson
    pearson_baseline
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np

from six.moves import range
from six import iteritems

def variance_weights(n_x, yr):
    """Calculating the variance for each object.

    If the model is ```user_based```, we calculate the variance_weights for the items,
    Otherwise we calculate them for the users.
    
    Only known ratings are taken into account. The variance for is defined as:
    .. math::
        \\text{var\\textsubscript{i}} = \\frac{\\limits_{i} (r_{u, i} - \mu_{u}) ^ 2}{n - 1}
    
    or

    .. math::
        \\text{var\\textsubscript{i}} = \\frac{\\limits_{i} (r_{i, v} - \mu_{v}) ^ 2}{n - 1}

    The variance-weight is defined as:

    .. math::
        \\text{var\\texsubscript{i}} = \\text{var\\textsubscript{i}} - 
        frac{\\text{var\\textsubscript{min}}}{\\text{var\\textsubscript{max}}}
    """
    cdef np.ndarray[np.double_t, ndim=1] var_weights
    var_weights = np.zeros((n_x), np.double)

    for y, y_ratings in iteritems(yr):
        var_weights[y] = np.var([rating for (iid, rating) in y_ratings])

    minMax = np.min(var_weights) / np.max(var_weights)

    return var_weights, minMax



def cosine(n_x, yr, n_y, min_support, significance_weighting=False, significance_beta=50,
           variance_weighting=False, inverse_user_frequency=False, case_amplification=False,
           p=1):
    """Compute the cosine similarity between all pairs of users (or items).

    Only **common** users (or items) are taken into account. The cosine
    similarity is defined as:

    .. math::
        \\text{cosine_sim}(u, v) = \\frac{
        \\sum\\limits_{i \in I_{uv}} r_{ui} \cdot r_{vi}}
        {\\sqrt{\\sum\\limits_{i \in I_{uv}} r_{ui}^2} \cdot
        \\sqrt{\\sum\\limits_{i \in I_{uv}} r_{vi}^2}
        }

    or

    .. math::
        \\text{cosine_sim}(i, j) = \\frac{
        \\sum\\limits_{u \in U_{ij}} r_{ui} \cdot r_{uj}}
        {\\sqrt{\\sum\\limits_{u \in U_{ij}} r_{ui}^2} \cdot
        \\sqrt{\\sum\\limits_{u \in U_{ij}} r_{uj}^2}
        }

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).

    For details on cosine similarity, see on `Wikipedia
    <https://en.wikipedia.org/wiki/Cosine_similarity#Definition>`__.
    """

    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.int_t, ndim=2] prods
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim=2] sqi
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim=2] sqj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj, ri, rj
    cdef int min_sprt = min_support
    cdef double beta = significance_beta
    cdef int ny = n_y

    #inverse user frequency    
    cdef int xa
    cdef double f, nom
    cdef np.ndarray[np.double_t, ndim=2] sum_fxai
    cdef np.ndarray[np.double_t, ndim=2] sum_fxa
    cdef np.ndarray[np.double_t, ndim=2] sum_fxi
    cdef np.ndarray[np.double_t, ndim=2] sum_fxa_squared
    cdef np.ndarray[np.double_t, ndim=2] sum_fxi_squared
    cdef np.ndarray[np.double_t, ndim=2] sum_sum_fxai
    cdef np.ndarray[np.double_t, ndim=2] u
    cdef np.ndarray[np.double_t, ndim=2] v


    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    if not inverse_user_frequency:
        prods = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.int)
        sqj = np.zeros((n_x, n_x), np.int)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += ri * rj
                    sqi[xi, xj] += ri**2
                    sqj[xi, xj] += rj**2
               

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] < min_sprt:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = prods[xi, xj] / denum
                    if significance_weighting:
                        sim[xi, xj] *= (freq[xi, xj] / beta)
        
                sim[xj, xi] = sim[xi, xj]

    else:
        sum_fxai = np.zeros((n_x, n_x), np.double)
        sum_fxa = np.zeros((n_x, n_x), np.double)
        sum_fxi = np.zeros((n_x, n_x), np.double)
        sum_fxa_squared = np.zeros((n_x, n_x), np.double)
        sum_fxi_squared = np.zeros((n_x, n_x), np.double)
        for y, y_ratings in iteritems(yr):
            for xa, ra in y_ratings:
                for xi, ri, in y_ratings:
                    freq[xi, xj] += 1
                    f = np.log(ny / len(y_ratings))
                    sum_fxai[xa, xi] += f * ra * ri
                    sum_fxa[xa, xi] += f * ra
                    sum_fxi[xa, xi] += f * ri
                    sum_fxa_squared[xa, xi] += f * ra ** 2
                    sum_fxi_squared[xa, xi] += f * ri ** 2
    
        sum_sum_fxai = np.zeros((n_x, n_x), np.double)
        u = np.zeros((n_x, n_x), np.double)
        v = np.zeros((n_x, n_x), np.double)
        for y, y_ratings in iteritems(yr):
            for xa in range(n_x):
                for xi in range(xa + 1, n_x):
                    f = np.log(ny / len(y_ratings))
                    sum_sum_fxai[xa, xi] += f * sum_fxai[xa, xi]
                    u[xa, xi] += f * (sum_fxa_squared[xa, xi] - sum_fxa[xa, xi] ** 2)
                    v[xa, xi] += f * (sum_fxi_squared[xa, xi] - sum_fxi[xa, xi] ** 2)

            sum_sum_fxai[xi, xa] = sum_sum_fxai[xa, xi]
            u[xi, xa] = v[xa, xi]
            v[xi, xa] = u[xa, xi]
                
            
        for xa in range(n_x):
            sim[xa, xa] = 1
            for xi in range(xa + 1, n_x):
                if freq[xa, xi] < min_sprt:
                    sim[xa, xi] = 0
                else:
                    nom = sum_sum_fxai[xa, xi] - (sum_fxa[xa, xi] * sum_fxi[xa, xi])
                    sim[xa, xi] = nom / np.sqrt(u[xa, xi] * v[xa, xi])
                    if case_amplification:
                        sim[xa, xi] *= sim[xa, xi] ** p
    
            sim[xi, xa] = sim[xa, xi]
    
    return sim

def mad(n_x, yr, n_y, min_support, significance_weighting=False, significance_beta=50,
        variance_weighting=False, var_weight_yr=None, inverse_user_frequency=False,
        case_amplification=False, p=1):
    """Compute the Mean Absolute Difference similarity between all pairs of 
    users (or items).

    Only **common** users (or items) are takin into account. The Mean Absolute 
    Difference is defined as:

    .. math ::
        \text{mad}(u, v) = \\frac{1}{|I_{uv}|} \cdot
        \\sum\\limits_{i \in I{uv}} (r_{ui} - r_{vi})^2

    or

    .. math ::
        \text{mad}(i, j) = \\frac{1}{|U_{ij}|} \cdot
        \\sum\\limits_{u \in U{ij}} (r_{ui} - r_{uj})^2

    ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).

    The MAD-similarity is then defined as:
 
    .. math ::
        \\text{mad_sim}(u, v) &= \\frac{1}{\\text{mad}(u, v) + 1}\\\\
        \\text{mad_sim}(i, j) &= \\frac{1}{\\text{mad}(i, j) + 1}

    The :math:`+ 1` term is just here to avoid dividing by zero.


    For details on MAD, see third definition on `Wikipedia
    <https://en.wikipedia.org/wiki/Mean_absolute_difference#Calculation>`__.

    """

    # sum |r_xy - r_x'y| for common ys
    cdef np.ndarray[np.double_t, ndim=2] abs_diff
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim
    # the variance weighting matrix
    cdef np.ndarray[np.double_t, ndim=1] var_weights
    # the sum of variance weights
    cdef np.ndarray[np.double_t, ndim=1] sum_var_weights

    cdef int xi, xj, ri, rj
    cdef int min_sprt = min_support
    cdef double beta = significance_beta
    cdef double minMax, diff, mad
    cdef int ny = n_y

    if variance_weighting:
        var_weights, minMax = variance_weights(n_y, var_weight_yr)


    abs_diff = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)
    sum_var_weights = np.zeros((n_x), np.double)

    #this might have to be changed when we get to default voting, let's leave it for now
    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                diff = np.abs(ri - rj)
                if inverse_user_frequency:
                    diff *= np.log(ny / len(y_ratings))
                if variance_weighting:
                    abs_diff[xi, xj] += diff * var_weights[y]
                    sum_var_weights[y] += var_weights[y]
                else:
                    abs_diff[xi, xj] += diff
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] == 0
            else:
                # inverse of (mad + 1) (+ 1 to avoid dividing by zero)
                sim[xi, xj] = 1 / (abs_diff[xi, xj] / freq[xi, xj] + 1)
                if significance_weighting:
                    sim[xi, xj] *= (freq[xi, xj] / beta)
                if variance_weighting:
                    sim[xi, xj] *= 1 / sum_var_weights[xi]
                if case_amplification:
                    sim[xi, xj] *= sim[xi, xj] ** p
            sim[xj, xi] = sim[xi, xj]

    return sim

def msd(n_x, yr, n_y, min_support, significance_weighting=False, significance_beta=50,
        variance_weighting=False, var_weight_yr=None, inverse_user_frequency=False,
        case_amplification=False, p=1):
    """Compute the Mean Squared Difference similarity between all pairs of
    users (or items).

    Only **common** users (or items) are taken into account. The Mean Squared
    Difference is defined as:

    .. math ::
        \\text{msd}(u, v) = \\frac{1}{|I_{uv}|} \cdot
        \\sum\\limits_{i \in I_{uv}} (r_{ui} - r_{vi})^2

    or

    .. math ::
        \\text{msd}(i, j) = \\frac{1}{|U_{ij}|} \cdot
        \\sum\\limits_{u \in U_{ij}} (r_{ui} - r_{uj})^2

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).

    The MSD-similarity is then defined as:

    .. math ::
        \\text{msd_sim}(u, v) &= \\frac{1}{\\text{msd}(u, v) + 1}\\\\
        \\text{msd_sim}(i, j) &= \\frac{1}{\\text{msd}(i, j) + 1}

    The :math:`+ 1` term is just here to avoid dividing by zero.


    For details on MSD, see third definition on `Wikipedia
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation#Formula>`__.

    """

    # sum (r_xy - r_x'y)**2 for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq_diff
    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim
    # the variance weighting matrix
    cdef np.ndarray[np.double_t, ndim=1] var_weights
    # the sum of variance weights
    cdef np.ndarray[np.double_t, ndim=1] sum_var_weights

    cdef int xi, xj, ri, rj
    cdef int min_sprt = min_support
    cdef double beta = significance_beta
    cdef double minMax, diff, msd
    cdef int ny = n_y

    sq_diff = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)
    sum_var_weights = np.zeros((ny), np.double)

    if variance_weighting:
        var_weights, minMax = variance_weights(n_y, var_weight_yr)


    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                diff = (ri - rj) ** 2
                if inverse_user_frequency:
                    diff *= np.log(ny / len(y_ratings))
                if variance_weighting:
                    sq_diff[xi, xj] += diff * var_weights[y]
                    sum_var_weights[y] += var_weights[y]
                else:
                    sq_diff[xi, xj] += diff
                freq[xi, xj] += 1


    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] == 0
            else:
                # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)
                if significance_weighting:
                    sim[xi, xj] *= (freq[xi, xj] / beta)
                if variance_weighting:
                    sim[xi, xj] *= 1 / sum_var_weights[xi]
                if case_amplification:
                    sim[xi, xj] *= sim[xi, xj] ** p
            sim[xj, xi] = sim[xi, xj]

    return sim


def pearson(n_x, yr, n_y, min_support, significance_weighting=False, significance_beta=50,
            variance_weighting=False, inverse_user_frequency=False, case_amplification=False,
            p=1):
    """Compute the Pearson correlation coefficient between all pairs of users
    (or items).

    Only **common** users (or items) are taken into account. The Pearson
    correlation coefficient can be seen as a mean-centered cosine similarity,
    and is defined as:

    .. math ::
        \\text{pearson_sim}(u, v) = \\frac{ \\sum\\limits_{i \in I_{uv}}
        (r_{ui} -  \mu_u) \cdot (r_{vi} - \mu_{v})} {\\sqrt{\\sum\\limits_{i
        \in I_{uv}} (r_{ui} -  \mu_u)^2} \cdot \\sqrt{\\sum\\limits_{i \in
        I_{uv}} (r_{vi} -  \mu_{v})^2} }

    or

    .. math ::
        \\text{pearson_sim}(i, j) = \\frac{ \\sum\\limits_{u \in U_{ij}}
        (r_{ui} -  \mu_i) \cdot (r_{uj} - \mu_{j})} {\\sqrt{\\sum\\limits_{u
        \in U_{ij}} (r_{ui} -  \mu_i)^2} \cdot \\sqrt{\\sum\\limits_{u \in
        U_{ij}} (r_{uj} -  \mu_{j})^2} }

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).


    Note: if there are no common users or items, similarity will be 0 (and not
    -1).

    For details on Pearson coefficient, see `Wikipedia
    <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample>`__.

    """

    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.int_t, ndim=2] prods
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim=2] sqi
    # sum (rx'y ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim=2] sqj
    # sum (rxy) for common ys
    cdef np.ndarray[np.int_t, ndim=2] si
    # sum (rx'y) for common ys
    cdef np.ndarray[np.int_t, ndim=2] sj
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef int xi, xj, ri, rj
    cdef int min_sprt = min_support
    cdef double beta = significance_beta

    freq = np.zeros((n_x, n_x), np.int)
    prods = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.int)
    sqj = np.zeros((n_x, n_x), np.int)
    si = np.zeros((n_x, n_x), np.int)
    sj = np.zeros((n_x, n_x), np.int)
    sim = np.zeros((n_x, n_x), np.double)

    for y, y_ratings in iteritems(yr):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                prods[xi, xj] += ri * rj
                freq[xi, xj] += 1
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] == 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                                (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum
                    if significance_weighting:
                        sim[xi, xj] *= (freq[xi, xj] / beta)
                    if case_amplification:
                        sim[xi, xj] *= sim[xi, xj] ** p
            sim[xj, xi] = sim[xi, xj]

    return sim


def pearson_baseline(n_x, yr, n_y, min_support, global_mean, x_biases, y_biases,
                     shrinkage=100, significance_weighting=False, variance_weighting=False,
                     inverse_user_frequency=False, case_amplification=False, p=1):
    """Compute the (shrunk) Pearson correlation coefficient between all pairs
    of users (or items) using baselines for centering instead of means.

    The shrinkage parameter helps to avoid overfitting when only few ratings
    are available (see :ref:`similarity_measures_configuration`).

    The Pearson-baseline correlation coefficient is defined as:

    .. math::
        \\text{pearson_baseline_sim}(u, v) = \hat{\\rho}_{uv} = \\frac{
            \\sum\\limits_{i \in I_{uv}} (r_{ui} -  b_{ui}) \cdot (r_{vi} -
            b_{vi})} {\\sqrt{\\sum\\limits_{i \in I_{uv}} (r_{ui} -  b_{ui})^2}
            \cdot \\sqrt{\\sum\\limits_{i \in I_{uv}} (r_{vi} -  b_{vi})^2}}

    or

    .. math::
        \\text{pearson_baseline_sim}(i, j) = \hat{\\rho}_{ij} = \\frac{
            \\sum\\limits_{u \in U_{ij}} (r_{ui} -  b_{ui}) \cdot (r_{uj} -
            b_{uj})} {\\sqrt{\\sum\\limits_{u \in U_{ij}} (r_{ui} -  b_{ui})^2}
            \cdot \\sqrt{\\sum\\limits_{u \in U_{ij}} (r_{uj} -  b_{uj})^2}}

    The shrunk Pearson-baseline correlation coefficient is then defined as:

    .. math::
        \\text{pearson_baseline_shrunk_sim}(u, v) &= \\frac{|I_{uv}| - 1}
        {|I_{uv}| - 1 + \\text{shrinkage}} \\cdot \hat{\\rho}_{uv}

        \\text{pearson_baseline_shrunk_sim}(i, j) &= \\frac{|U_{ij}| - 1}
        {|U_{ij}| - 1 + \\text{shrinkage}} \\cdot \hat{\\rho}_{ij}


    Obviously, a shrinkage parameter of 0 amounts to no shrinkage at all.

    Note: here again, if there are no common users/items, similarity will be 0
    (and not -1).

    Motivations for such a similarity measure can be found on the *Recommender
    System Handbook*, section 5.4.1.
    """

    # number of common ys
    cdef np.ndarray[np.int_t, ndim=2] freq
    # sum (r_xy - b_xy) * (r_x'y - b_x'y) for common ys
    cdef np.ndarray[np.double_t, ndim=2] prods
    # sum (r_xy - b_xy)**2 for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq_diff_i
    # sum (r_x'y - b_x'y)**2 for common ys
    cdef np.ndarray[np.double_t, ndim=2] sq_diff_j
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim=2] sim

    cdef np.ndarray[np.double_t, ndim=1] x_biases_
    cdef np.ndarray[np.double_t, ndim=1] y_biases_

    cdef int xi, xj
    cdef double ri, rj, diff_i, diff_j, partial_bias
    cdef int min_sprt = min_support
    cdef double global_mean_ = global_mean

    freq = np.zeros((n_x, n_x), np.int)
    prods = np.zeros((n_x, n_x), np.double)
    sq_diff_i = np.zeros((n_x, n_x), np.double)
    sq_diff_j = np.zeros((n_x, n_x), np.double)
    sim = np.zeros((n_x, n_x), np.double)

    x_biases_ = x_biases
    y_biases_ = y_biases

    # Need this because of shrinkage. When pearson coeff is zero when support
    # is 1, so that's OK.
    min_sprt = max(2, min_sprt)

    for y, y_ratings in iteritems(yr):
        partial_bias = global_mean_ + y_biases_[y]
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                diff_i = (ri - (partial_bias + x_biases_[xi]))
                diff_j = (rj - (partial_bias + x_biases_[xj]))
                prods[xi, xj] += diff_i * diff_j
                sq_diff_i[xi, xj] += diff_i**2
                sq_diff_j[xi, xj] += diff_j**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                sim[xi, xj] = prods[xi, xj] / (np.sqrt(sq_diff_i[xi, xj] *
                                                       sq_diff_j[xi, xj]))
                # the shrinkage part
                sim[xi, xj] *= (freq[xi, xj] - 1) / (freq[xi, xj] - 1 +
                                                     shrinkage)
                if case_amplification:
                    sim[xi, xj] *= sim[xi, xj] ** p
            sim[xj, xi] = sim[xi, xj]

    return sim
