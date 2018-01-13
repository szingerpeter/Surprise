"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
from six.moves import range

from .algo_base import AlgoBase
from .predictions import PredictionImpossible
from ..utils import get_rng


class SVD(AlgoBase):
    """The famous *SVD* algorithm, as popularized by `Simon Funk
    <http://sifter.org/~simon/journal/20061211.html>`_ during the Netflix
    Prize. When baselines are not used, this is equivalent to Probabilistic
    Matrix Factorization :cite:`salakhutdinov2008a` (see :ref:`note
    <unbiased_note>` below).

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_i` and :math:`q_i`.

    For details, see equation (5) from :cite:`Koren:2009`. See also
    :cite:`Ricci:2010`, section 5.3.1.

    To estimate all the unknown, we minimize the following regularized squared
    error:

    .. math::
        \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)


    The minimization is performed by a very straightforward stochastic gradient
    descent:

    .. math::
        b_u &\\leftarrow b_u &+ \gamma (e_{ui} - \lambda b_u)\\\\
        b_i &\\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\\\
        p_u &\\leftarrow p_u &+ \gamma (e_{ui} \\cdot q_i - \lambda p_u)\\\\
        q_i &\\leftarrow q_i &+ \gamma (e_{ui} \\cdot p_u - \lambda q_i)

    where :math:`e_{ui} = r_{ui} - \\hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epochs`` times.
    Baselines are initialized to ``0``. User and item factors are randomly
    initialized according to a normal distribution, which can be tuned using
    the ``init_mean`` and ``init_std_dev`` parameters.

    You also have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization terms are set to ``0.02``.

    .. _unbiased_note:

    .. note::
        You can choose to use an unbiased version of this algorithm, simply
        predicting:

        .. math::
            \hat{r}_{ui} = q_i^Tp_u

        This is equivalent to Probabilistic Matrix Factorization
        (:cite:`salakhutdinov2008a`, section 2) and can be achieved by setting
        the ``biased`` parameter to ``False``.


    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        biased(bool): Whether to use biases. See :ref:`note
            <unbiased_note>` above.  Default is ``True``.
		mean_centered(bool): Whether to mean center ratings. Default is ``True``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.005``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.
<<<<<<< HEAD
        amau(bool): Whether to treat missing values as unkown (``True``) or negative (``False``)
        missing_val: If AMAN, what value to assign to missing values
        downweight: How much downweight the negatively treated missing values
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, mean_centered=True, init_mean=0, init_std_dev=.1, lr_all=.005, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, verbose=False, amau=True, missing_val=0, downweight=.001):
=======

    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):
>>>>>>> upstream/master

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.mean_centered = mean_centered
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.amau = amau
        self.missing_val = missing_val
        self.downweight = downweight

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # OK, let's breath. I've seen so many different implementation of this
        # algorithm that I just not sure anymore of what it should do. I've
        # implemented the version as described in the BellKor papers (RS
        # Handbook, etc.). Mymedialite also does it this way. In his post
        # however, Funk seems to implicitly say that the algo looks like this
        # (see reg below):
        # for f in range(n_factors):
        #       for _ in range(n_iter):
        #           for u, i, r in all_ratings:
        #               err = r_ui - <p[u, :f+1], q[i, :f+1]>
        #               update p[u, f]
        #               update q[i, f]
        # which is also the way https://github.com/aaw/IncrementalSVD.jl
        # implemented it.
        #
        # Funk: "Anyway, this will train one feature (aspect), and in
        # particular will find the most prominent feature remaining (the one
        # that will most reduce the error that's left over after previously
        # trained features have done their best). When it's as good as it's
        # going to get, shift it onto the pile of done features, and start a
        # new one. For efficiency's sake, cache the residuals (all 100 million
        # of them) so when you're training feature 72 you don't have to wait
        # for predictRating() to re-compute the contributions of the previous
        # 71 features. You will need 2 Gig of ram, a C compiler, and good
        # programming habits to do this."

        # A note on cythonization: I haven't dived into the details, but
        # accessing 2D arrays like pu using just one of the indices like pu[u]
        # is not efficient. That's why the old (cleaner) version can't be used
        # anymore, we need to compute the dot products by hand, and update
        # user and items factors by iterating over all factors...

        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i
        cdef double r
        cdef double global_mean = self.trainset.global_mean
<<<<<<< HEAD
        cdef double missing_val = self.missing_val
        cdef double downweight_rating
		
=======

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

>>>>>>> upstream/master
        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        
        if not self.mean_centered:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            if self.amau:
                downweight_rating = 1
                for u, i, r in trainset.all_ratings():
                    pu[u], qi[i], bu[u], bi[i] = self.update(pu[u], qi[i], bu[u], bi[i], global_mean, r, downweight_rating)               

            else:
                for u in trainset.all_users():
                    for i in trainset.all_items():
                        downweight_rating = 1

                        rating = trainset.get_rating(u, i)
                        if rating != None:
                            r = rating
                        else:
                            r = missing_val
                            downweight_rating *= self.downweight
                        pu[u], qi[i], bu[u], bi[i] = self.update(pu[u], qi[i], bu[u], bi[i], global_mean, r, downweight_rating)               	


        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
		
        est = 0
        if self.mean_centered:
            est += self.trainset.global_mean
		
        if self.biased:
            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est

    def update(self, puu, qii, buu, bii, train_global_mean, rating, downweight_rating):
        cdef int f
        cdef double err, dot, est
    
        cdef double global_mean = train_global_mean
        cdef np.ndarray[np.double_t, ndim=1] pu_u = puu
        cdef np.ndarray[np.double_t, ndim=1] qi_i = qii
        cdef double bu_u = buu
        cdef double bi_i = bii
    
        cdef double r = rating
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
    
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double downweight = downweight_rating
        
        dot = 0
        for f in range(self.n_factors):
            dot += qii[f] * puu[f]
        err = (r - (global_mean + buu + bii + dot))
        #update biases    
        if self.biased:
            bu_u += lr_bu * downweight * (err - reg_bu * bu_u)
            bi_i += lr_bi * downweight * (err - reg_bi * bi_i)
        
        #update factors
        for f in range(self.n_factors):
            pu_u_f = pu_u[f]
            qi_i_f = qi_i[f]
            pu_u[f] = lr_pu * downweight * (err * qi_i_f - reg_pu * pu_u_f)
            qi_i[f] = lr_qi * downweight * (err * pu_u_f - reg_qi * qi_i_f)

        return pu_u, qi_i, bu_u, bi_i
    


class SVDpp(AlgoBase):
    """The *SVD++* algorithm, an extension of :class:`SVD` taking into account
    implicit ratings.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\\left(p_u +
        |I_u|^{-\\frac{1}{2}} \sum_{j \\in I_u}y_j\\right)

    Where the :math:`y_j` terms are a new set of item factors that capture
    implicit ratings. Here, an implicit rating describes the fact that a user
    :math:`u` rated an item :math:`j`, regardless of the rating value.

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_i`, :math:`q_i` and :math:`y_i`.


    For details, see section 4 of :cite:`Koren:2008:FMN`. See also
    :cite:`Ricci:2010`, section 5.3.1.

    Just as for :class:`SVD`, the parameters are learned using a SGD on the
    regularized squared error objective.

    Baselines are initialized to ``0``. User and item factors are randomly
    initialized according to a normal distribution, which can be tuned using
    the ``init_mean`` and ``init_std_dev`` parameters.

    You have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization terms are set to ``0.02``.

    Args:
        n_factors: The number of factors. Default is ``20``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
		biased(bool): Whether to use biases. See :ref:`note
            <unbiased_note>` above.  Default is ``True``.
		mean_centered(bool): Whether to mean center ratings. Default is ``True``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.007``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_yj: The learning rate for :math:`y_j`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_yj: The regularization term for :math:`y_j`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.
<<<<<<< HEAD
        amau(bool): Whether to treat missing values as unkown (``True``) or negative (``False``)
        missing_val: If AMAN, what value to assign to missing values
        downweight: How much downweight the negatively treated missing values
    """

    def __init__(self, n_factors=20, n_epochs=20, biased=True, mean_centered=True, init_mean=0, init_std_dev=.1, lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None, verbose=False, amau=True, missing_val=0, downweight=.001):
=======

    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        yj(numpy array of size (n_items, n_factors)): The (implicit) item
            factors (only exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_yj=None, random_state=None, verbose=False):
>>>>>>> upstream/master

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.mean_centered = mean_centered
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.amau = amau
        self.missing_val = missing_val
        self.downweight = downweight

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi
        # item implicit factors
        cdef np.ndarray[np.double_t, ndim=2] yj

        cdef int u, i, j
        cdef double _, r
        cdef double global_mean = self.trainset.global_mean
        cdef double missing_val = self.missing_val
        cdef double downweight_rating

		
        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

<<<<<<< HEAD
        pu = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_users, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))
        yj = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))
							  
        if not self.mean_centered:
            global_mean = 0
=======
        rng = get_rng(self.random_state)

        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        yj = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors, np.double)
>>>>>>> upstream/master

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            if self.amau:
                downweight_rating = 1
                for u, i, r in trainset.all_ratings():

                    # items rated by u. This is COSTLY
                    Iu = [j for (j, _) in trainset.ur[u]]
                    pu[u], qi[i], yj, bu[u], bi[i] = self.update(pu[u], qi[i], Iu, yj, bu[u], bi[i], global_mean, r, downweight_rating)   

            else:
                for u in trainset.all_users():
                    for i in trainset.all_items():
                        downweight_rating = 1
                        # items rated by u. This is COSTLY
                        Iu = [j for (j, _) in trainset.ur[u]]

                        rating = trainset.get_rating(u, i)
                        if rating != None:
                            r = rating
                        else:
                            r = missing_val
                            downweight_rating *= self.downweight
                        pu[u], qi[i], yj_updated, bu[u], bi[i] = self.update(pu[u], qi[i], Iu, yj, bu[u], bi[i], global_mean, r, downweight_rating)
                        for j in Iu:
                            yj[j] = yj_updated[j]
             
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj


    def estimate(self, u, i):

        est = 0
		
        if self.mean_centered:
            est += self.trainset.global_mean

        if self.biased:
            if self.trainset.knows_user(u):
                est += self.bu[u]

            if self.trainset.knows_item(i):
                est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est

    def update(self, puu, qii, iu, yj, buu, bii, train_global_mean, rating, downweight_rating):
        cdef int f
        cdef double err, dot, est
    
        cdef double global_mean = train_global_mean
        cdef np.ndarray[np.double_t] u_impl_fdb
        cdef np.ndarray[np.double_t, ndim=1] pu_u = puu
        cdef np.ndarray[np.double_t, ndim=1] qi_i = qii
        cdef list Iu = iu
        cdef np.ndarray[np.double_t, ndim=2] Yj = yj
        cdef double bu_u = buu
        cdef double bi_i = bii

    
        cdef double r = rating
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj
    
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj
        cdef double downweight = downweight_rating
        
        sqrt_Iu = np.sqrt(len(Iu))

        # compute user implicit feedback
        u_impl_fdb = np.zeros(self.n_factors, np.double)
        for j in Iu:
            for f in range(self.n_factors):
                u_impl_fdb[f] += Yj[j, f] / sqrt_Iu

        # compute current error
        dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
        for f in range(self.n_factors):
            dot += qi_i[f] * (pu_u[f] + u_impl_fdb[f])

        err = (r - (global_mean + buu + bii + dot))
        
		#update biases    
        if self.biased:
            bu_u += lr_bu * downweight * (err - reg_bu * bu_u)
            bi_i += lr_bi * downweight * (err - reg_bi * bi_i)
        
        #update factors
        for f in range(self.n_factors):
            pu_u_f = pu_u[f]
            qi_i_f = qi_i[f]
            pu_u[f] = lr_pu * downweight * (err * qi_i_f - reg_pu * pu_u_f)
            qi_i[f] = lr_qi * downweight * (err * pu_u_f - reg_qi * qi_i_f)
            
            for j in Iu:
                Yj[j, f] += lr_yj * (err * qi_i_f / sqrt_Iu -
                                             reg_yj * Yj[j, f])

        return pu_u, qi_i, Yj, bu_u, bi_i



class NMF(AlgoBase):
    """A collaborative filtering algorithm based on Non-negative Matrix
    Factorization.

    This algorithm is very similar to :class:`SVD`. The prediction
    :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = q_i^Tp_u,

    where user and item factors are kept **positive**. Our implementation
    follows that suggested in :cite:`NMF:2014`, which is equivalent to
    :cite:`Zhang96` in its non-regularized form. Both are direct applications
    of NMF for dense matrices :cite:`NMF_algo`.

    The optimization procedure is a (regularized) stochastic gradient descent
    with a specific choice of step size that ensures non-negativity of factors,
    provided that their initial values are also positive.

    At each step of the SGD procedure, the factors :math:`f` or user :math:`u`
    and item :math:`i` are updated as follows:

    .. math::
        p_{uf} &\\leftarrow p_{uf} &\cdot \\frac{\\sum_{i \in I_u} q_{if}
        \\cdot r_{ui}}{\\sum_{i \in I_u} q_{if} \\cdot \\hat{r_{ui}} +
        \\lambda_u |I_u| p_{uf}}\\\\
        q_{if} &\\leftarrow q_{if} &\cdot \\frac{\\sum_{u \in U_i} p_{uf}
        \\cdot r_{ui}}{\\sum_{u \in U_i} p_{uf} \\cdot \\hat{r_{ui}} +
        \lambda_i |U_i| q_{if}}\\\\

    where :math:`\lambda_u` and :math:`\lambda_i` are regularization
    parameters.

    This algorithm is highly dependent on initial values. User and item factors
    are uniformly initialized between ``init_low`` and ``init_high``. Change
    them at your own risks!

    A biased version is available by setting the ``biased`` parameter to
    ``True``. In this case, the prediction is set as

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u,

    still ensuring positive factors. Baselines are optimized in the same way as
    in the :class:`SVD` algorithm. While yielding better accuracy, the biased
    version seems highly prone to overfitting so you may want to reduce the
    number of factors (or increase regularization).

    Args:
        n_factors: The number of factors. Default is ``15``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``50``.
        biased(bool): Whether to use biases. Default is
            ``False``.
		mean_centered(bool): Whether to mean center ratings. Default is ``True``.
        reg_pu: The regularization term for users :math:`\lambda_u`. Default is
            ``0.06``.
        reg_qi: The regularization term for items :math:`\lambda_i`. Default is
            ``0.06``.
        reg_bu: The regularization term for :math:`b_u`. Only relevant for
            biased version. Default is ``0.02``.
        reg_bi: The regularization term for :math:`b_i`. Only relevant for
            biased version. Default is ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Only relevant for biased
            version. Default is ``0.005``.
        lr_bi: The learning rate for :math:`b_i`. Only relevant for biased
            version. Default is ``0.005``.
        init_low: Lower bound for random initialization of factors. Must be
            greater than ``0`` to ensure non-negative factors. Default is
            ``0``.
        init_high: Higher bound for random initialization of factors. Default
            is ``1``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.

    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

<<<<<<< HEAD
    def __init__(self, n_factors=15, n_epochs=50, biased=False, mean_centered=True, reg_pu=.06, reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005, init_low=0, init_high=1, verbose=False, amau=True, missing_val=0, downweight=.001):
=======
    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None, verbose=False):
>>>>>>> upstream/master

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.mean_centered = mean_centered
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose
        self.amau = amau
        self.missing_val = missing_val
        self.downweight = downweight

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user and item factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi

        # user and item biases
        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi

        # auxiliary matrices used in optimization process
        cdef np.ndarray[np.double_t, ndim=2] user_num
        cdef np.ndarray[np.double_t, ndim=2] user_denom
        cdef np.ndarray[np.double_t, ndim=2] item_num
        cdef np.ndarray[np.double_t, ndim=2] item_denom

        cdef int u, i
        cdef double r
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double global_mean = self.trainset.global_mean
        cdef double missing_val = self.missing_val
        cdef double downweight_rating
		
        # Randomly initialize user and item factors
        rng = get_rng(self.random_state)
        pu = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_users, self.n_factors))
        qi = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        if not self.mean_centered:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            user_num = np.zeros((trainset.n_users, self.n_factors))
            user_denom = np.zeros((trainset.n_users, self.n_factors))
            item_num = np.zeros((trainset.n_items, self.n_factors))
            item_denom = np.zeros((trainset.n_items, self.n_factors))

            if self.amau:
                downweight_rating = 1
                # Compute numerators and denominators for users and items factors
                for u, i, r in trainset.all_ratings():				 
                    user_num[u], user_denom[u], item_num[i], item_denom[i], bu[u], bi[i] = self.update(pu[u], qi[i], user_num[u], user_denom[u], item_num[i], item_denom[i], bu[u], bi[i], global_mean, r, downweight_rating)

            else:
                for u in trainset.all_users():
                    for i in trainset.all_items():
                        downweight_rating = 1                        

                        rating = trainset.get_rating(u, i)
                        if rating != None:
                            r = rating                            
                        else:
                            r = missing_val
                            downweight_rating *= self.downweight
                        user_num[u], user_denom[u], item_num[i], item_denom[i], bu[u], bi[i] = self.update(pu[u], qi[i], user_num[u], user_denom[u], item_num[i], item_denom[i], bu[u], bi[i], global_mean, r, downweight_rating)
            
            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(self.n_factors):
                    user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                    pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(self.n_factors):
                    item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                    qi[i, f] *= item_num[i, f] / item_denom[i, f]
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        est = 0
        if self.mean_centered:
            est += self.trainset.global_mean
			
        if self.biased:
            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est

    def update(self, puu, qii, u_num_u, u_denom_u, i_num_i, i_denom_i, buu, bii, train_global_mean, rating, downweight_rating):
        cdef int f
        cdef double err, dot, est
    
        cdef double global_mean = train_global_mean
        cdef np.ndarray[np.double_t, ndim=1] pu_u = puu
        cdef np.ndarray[np.double_t, ndim=1] qi_i = qii
        cdef np.ndarray[np.double_t, ndim=1] user_num_u = u_num_u
        cdef np.ndarray[np.double_t, ndim=1] user_denom_u = u_denom_u
        cdef np.ndarray[np.double_t, ndim=1] item_num_i = i_num_i,
        cdef np.ndarray[np.double_t, ndim=1] item_denom_i = i_denom_i        
        cdef double bu_u = buu
        cdef double bi_i = bii

    
        cdef double r = rating
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
    
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double downweight = downweight_rating


        # compute current estimation and error
        dot = 0  # <q_i, p_u>
        for f in range(self.n_factors):
            dot += qi_i[f] * pu_u[f]
        est = global_mean + bu_u + bi_i + dot
        err = r - est

        # update biases
        if self.biased:
            bu_u += lr_bu * downweight * (err - reg_bu * bu_u)
            bi_i += lr_bi * downweight * (err - reg_bi * bi_i)

        # compute numerators and denominators
        for f in range(self.n_factors):
            user_num_u[f] += qi_i[f] * r
            user_denom_u[f] += qi_i[f] * est
            item_num_i[f] += pu_u[f] * r
            item_denom_i[f] += pu_u[f] * est

        return user_num_u, user_denom_u, item_num_i, item_denom_i, bu_u, bi_i
