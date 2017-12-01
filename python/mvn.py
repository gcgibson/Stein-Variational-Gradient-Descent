import numpy as np
import numpy.matlib as nm
from svgd import SVGD


class MVN:
    def __init__(self, mu, A, time_series):
        self.mu = mu
        self.A = A
	self.time_series = time_series    
    def dlnprob(self, theta,t):
        prob = []
	prob.append( -1*(theta[:,0]-0)/A)
	for i in range(1,len(time_series)):
		prob.append((theta[:,i]-theta[:,i-1])/A + (time_series[t] - theta[:,i])/.0001)
	prob = np.array(prob)
	return np.transpose(prob)
if __name__ == '__main__':
    A = np.array([.01])
    mu = np.array([2])
    time_series = np.ones(1000)
    model = MVN(mu, A,time_series)
    
    x0 = np.random.normal(0,1, [100,len(time_series)]);
    theta = SVGD().update(x0,0, model.dlnprob, n_iter=1000, stepsize=0.01)
    # this is a sample from the first time step posterior 
    
    print (np.mean(theta,axis=0))
 
