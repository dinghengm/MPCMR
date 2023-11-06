import numpy as np
np.seterr(all='ignore')


import contextlib,  sys
from io import StringIO
from tqdm.auto import tqdm # progress bar
@contextlib.contextmanager
def nostdout():

    '''Prevent print to stdout, but if there was an error then catch it and
    print the output before raising the error.'''

    saved_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    except Exception:
        saved_output = sys.stdout
        sys.stdout = saved_stdout
        print(saved_output.getvalue())
        raise
    sys.stdout = saved_stdout

class T1_fitter(object):

    def __init__(self, ti_vec, t1res=1, t1min=1, t1max=5000, fit_method='mag', ndel=4):
        '''
        ti_vec: vector of inversion times (len(ti_vec) == len(data)
        t1res: resolution of t1 grid-search (in milliseconds)
        t1min,t1max: min/max t1 for grid search (in milliseconds)
        '''
        self.fit_method = fit_method.lower()
        self.t1min = t1min
        self.t1max = t1max
        self.t1res = t1res
        self.ndel = ndel
        if self.fit_method=='nlspr' or self.fit_method=='mag' or self.fit_method=='nls':
            self.init_nls(ti_vec)
        else:
            self.ti_vec = np.array(ti_vec, dtype=np.float64)

    def init_nls(self, new_tis=None):
        if new_tis is not None:
            self.ti_vec = np.matrix(new_tis, dtype=np.float64)
        #else:
        #    self.ti_vec = np.matrix(self.ti_vec, dtype=np.float)
        n = self.ti_vec.size
        self.t1_vec = np.matrix(np.arange(self.t1min, self.t1max+self.t1res, self.t1res, dtype=np.float64))
        self.the_exp = np.exp(-self.ti_vec.T * np.matrix(1/self.t1_vec))
        self.exp_sum = 1. / n * self.the_exp.sum(0).T
        self.rho_norm_vec = np.sum(np.power(self.the_exp,2), 0).T - 1./n*np.power(self.the_exp.sum(0).T,2)

    def __call__(self, d):
        # Work-aropund for pickle's (and thus multiprocessing's) inability to map a class method.
        # See http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma
        if self.fit_method=='nlspr':
            return self.t1_fit_nlspr(d)
        elif self.fit_method=='mag':
            return self.t1_fit_magnitude(d)
        elif self.fit_method=='lm':
            return self.t1_fit_lm(d)
        elif self.fit_method=='nls':
            return self.t1_fit_nls(d)
        
    def t1_fit_lm(self, data):
        '''
        Finds estimates of T1, a, and b using multi-dimensional
        Levenberg-Marquardt algorithm. The model |c*(1-k*exp(-t/T1))|^2
        is used: only one phase term (c), and data are magnitude-squared.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,k,c,residual

        '''
        from scipy.optimize import leastsq
        # Make sure data is a 1d vector
        data = np.array(data.ravel())
        n = data.shape[0]

        # Initialize fit values:
        # T1 tarting value is hard-coded here (TODO: something better! Quick coarse grid search using nlspr?)
        # k should be around 1 - cos(flip_angle) = 2
        # |c| is set to the sqrt of the data at the longest TI
        max_val = (np.abs(data[np.argmax(self.ti_vec)]))
        x0 = np.array([900., 2., max_val])

        predicted = lambda t1,k,c,ti: np.abs( c*(1 - k * np.exp(-ti/t1)) ) ** 2
        residuals = lambda x,ti,y: y - np.sqrt(predicted(x[0], x[1], x[2], ti))
        #err = lambda x,ti,y: np.sum(np.abs(residuals(x,ti,y)))
        x,extra = leastsq(residuals, x0, args=(self.ti_vec.T,data))
        # NOTE: I tried minimize with two different bounded search algorithms (SLSQP and L-BFGS-B), but neither worked very well.
        # An unbounded leastsq fit with subsequent clipping of crazy fit values seems to be the fastest and most robust.
        #x0_bounds = [[0.,5000.],[None,None],[0.,max_val*10.]]
        #res = minimize(err, x0, args=(self.ti_vec.T,data), method='L-BFGS-B', bounds=x0_bounds, options={'disp':False, 'iprint':1, 'maxiter':100, 'ftol':1e-06})

        t1 = x[0].clip(self.t1min, self.t1max)
        k = x[1]
        c = x[2]

        # Compute the residual
        y_hat = predicted(t1, k, c, self.ti_vec)
        residual = 1. / np.sqrt(n) * np.sqrt(np.power(1 - y_hat / data.T, 2).sum())

        return(t1,k,c,residual)

    def t1_fit_nls(self, data):
        '''
        Finds estimates of T1, a, and b using a nonlinear least
        squares approach. The model a+b*exp(-t/T1) is used.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,b,a,residual

        Based on matlab code written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
         (c) Board of Trustees, Leland Stanford Junior University.
        See their 2010 MRM paper here: http://www.ncbi.nlm.nih.gov/pubmed/20564597.
        '''
        # Make sure data is a column vector
        data = np.matrix(data.ravel()).T
        n = data.shape[0]

        y_sum = data.sum()

        rho_ty_vec = (data.T * self.the_exp).T - self.exp_sum * y_sum
        # sum(theExp.^2, 1)' - 1/nlsS.N*(sum(theExp,1)').^2;

        # The maximizing criterion
        # [tmp,ind] = max( abs(rhoTyVec).^2./rhoNormVec );
        ind = np.argmax(np.power(np.abs(rho_ty_vec), 2)/self.rho_norm_vec)

        t1_hat = self.t1_vec[0,ind]
        b_hat = rho_ty_vec[ind,0] / self.rho_norm_vec[ind,0]
        a_hat = 1. / n * (y_sum - b_hat * self.the_exp[:,ind].sum())

        # Compute the residual
        model_val = a_hat + b_hat * np.exp(-self.ti_vec / t1_hat)
        residual = 1. / np.sqrt(n) * np.sqrt(np.power(1 - model_val / data.T, 2).sum())

        return(t1_hat,b_hat,a_hat,residual)


    def t1_fit_nlspr(self, data):
        '''
        Finds estimates of T1, a, and b using a nonlinear least
        squares approach. The model +-|aMag + bMag*exp(-t/T1)| is used.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,b,a,residual

        Based on matlab code written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
         (c) Board of Trustees, Leland Stanford Junior University
        '''
        data = np.matrix(data.ravel()).T
        n = data.shape[0]

        t1 = np.zeros(n)
        b = np.zeros(n)
        a = np.zeros(n)
        resid = np.zeros(n)
        for i in range(n):
            if i>0:
                data[i-1] = -data[i-1]
            (t1[i],b[i],a[i],resid[i]) = self.t1_fit_nls(data)

        ind = np.argmin(resid);

        return(t1[ind],b[ind],a[ind],resid[ind],ind)

    def t1_fit_magnitude(self, data):
        if self.ndel > 0 and self.ti_vec.size >= self.ndel + 4:
            indx = data.argmin()      # find the data point closest to the null
            indx_to_del = range(indx - int(np.floor(self.ndel/2)) + 1, indx + int(np.ceil(self.ndel/2)) + 1)
            if indx_to_del[0] >= 0 and indx_to_del[-1] < self.ti_vec.size:
                tis = np.delete(self.ti_vec, indx_to_del)
                data = np.delete(data, indx_to_del)
                for n in range(indx_to_del[0]):
                    data[n] = -data[n]
            else:
                tis = self.ti_vec
                for n in range(indx):
                    data[n] = -data[n]
            fit = T1_fitter(tis, fit_method='mag', t1min=self.t1min, t1max=self.t1max, t1res=self.t1res, ndel=self.ndel)
            (t1, b, a, res) = fit.t1_fit_nls(data)
        else:
            (t1, b, a, res, ind) = self.t1_fit_nlspr(data)
        return (t1, b, a, res)
    
def go_fit_T1(data=None,ti_vec=None,t1res=1,t1min=1,t1max=5000,err_method='mag',delete=4,jobs=4):
    import os
    import sys
    from multiprocessing import Pool
    
    print('Fitting T1 model')
    fit = T1_fitter(ti_vec,t1res,t1min,t1max,err_method,delete)

    if len(np.shape(data))==4:
        Nx,Ny,Nz,Nd=np.shape(data)
        NxNyNz=Nx*Ny*Nz
    else:
        Nx,Ny,Nd=np.shape(data)
        Nz=1
        NxNyNz=Nx*Ny*Nz
    update_step = 500
    update_interval = round(NxNyNz/float(update_step))

    data_tmp=np.reshape(data,(NxNyNz,Nd))
    t1_tmp=np.zeros(NxNyNz,dtype=np.float64)
    a_tmp=np.zeros(NxNyNz,dtype=np.float64)
    b_tmp=np.zeros(NxNyNz,dtype=np.float64)
    res_tmp=np.zeros(NxNyNz,dtype=np.float64)
    if jobs<2:
        for i in tqdm(range(NxNyNz)):
            d=data_tmp[i,:]
            t1_tmp[i],b_tmp[i],a_tmp[i],res_tmp[i]=fit(d)
    else:
        p=Pool(jobs)
        work=[data_tmp[i,:] for i in range(NxNyNz)]
        workers=p.map_async(fit,work)
        num_updates=0
        while not workers.ready():
            i = NxNyNz - workers._number_left * workers._chunksize
            if i>=update_interval*num_updates:
                num_updates+=1
                if num_updates<=update_step:
                    print('\r[{0}{1}] {2}%'.format('#'*num_updates, ' '*(update_step-num_updates), num_updates*5))
        
        out=workers.get()
        for i in range(NxNyNz):
            t1_tmp[i]=out[i][0]
            b_tmp[i]=out[i][1]
            a_tmp[i]=out[i][2]
            res_tmp[i]=out[i][3]
    t1_map=np.reshape(t1_tmp,(Nx,Ny,Nz))
    b_map=np.reshape(b_tmp,(Nx,Ny,Nz))
    a_map=np.reshape(a_tmp,(Nx,Ny,Nz))
    res_map=np.reshape(res_tmp,(Nx,Ny,Nz))
    print('finished.')
    return t1_map,b_map,a_map,res_map