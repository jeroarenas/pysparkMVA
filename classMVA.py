from pyspark.mllib.util import MLUtils
from pyspark.rdd import RDD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from sklearn.preprocessing import label_binarize
from pyspark.mllib.feature import StandardScaler
from pyspark.sql import Row
import urllib2
import pyspark.mllib.regression as mlreg
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
import numpy as np
import numpy as np
from numpy import linalg as lin
import math


class MVA(object):
  
    """
    
    This class solves the MVA methods for feature extraction

    """

    _typeMVA=None
    _typeReg=None
    _typeNorm=None
    _tol=None
    _numVariables=None
    _M =None
    _R= None
    _data=None #An RDD with format ([y0, y1, ..., yM], [x0, x1, ..., xN])
    _normdata = None
    _scaler=None
    _U=None
    _regParam=None 
    _step=None
    _iterations=None
    _max_Ustep=None
    _W=None
    
    def __init__(self, typeMVA, typeReg,typeNorm, tol,numFeatures ,regParam=0.01, step=1e-3, iterations=100, max_Ustep=10):
        
        """Class initializer
        :param typeMVA:
            Type of MVA method: PCA, OPLS or CCA
        :param typeReg:
            Type of Regularization used:
        :param typeNorm:
            Type of Normalization used:
        """

        self._typeMVA=typeMVA
        self._typeReg=typeReg
        self._regParam=regParam
        self._typeNorm=typeNorm
        self._tol=tol
        self._R=numFeatures
        self._step=step
        self._iterations=iterations
        self._max_Ustep=max_Ustep
        
        if typeMVA not in ['PCA', 'OPLS', 'CCA']:
            print 'The type of MVA is not correct'


    def prepareData(self, data):

        if data.filter(lambda x: not isinstance(x,LabeledPoint)).count() == 0:
            #Case 1: All points in dataset are LabeledPoints
            #Check if number of features in X is constant
            x_len = data.map(lambda x: len(Vectors.dense(x.features.toArray()))).cache()
            self._numVariables = x_len.first()
            if len(x_len.distinct().collect())!=1:
                print 'All feature vectors should have the same length. Aborting.'
                return False
            try:
                if self._typeMVA=='PCA':
                    self._data = (data.map(lambda x: Vectors.dense(x.features.toArray()))
                                      .map(lambda x: (x, x)))
                    
                    self._M = self._numVariables
                   
                    
                else:
                    set_classes = data.map(lambda x: x.label).distinct().collect()
                    
                    self._M = len(set_classes)
                    
                        
                    
                    
                    self._data = data.map(lambda x: (Vectors.dense(label_binarize([x.label], classes=set_classes).flatten()), 
                                            Vectors.dense(x.features.toArray())))
                return True
            except:
                return False
        
        elif data.filter(lambda x: not isinstance(x,tuple)).count() ==0:
            #Case 2: All points in dataset are tuples of numpy arrays
            try:
                x_len = data.map(lambda x: len(Vectors.dense(x[1]))).cache()
                self._numVariables = x_len.first()
                if len(x_len.distinct().collect())!=1:
                    print 'All feature vectors should have the same length. Aborting.'
                    return False
                y_len = data.map(lambda x: len(Vectors.dense(x[0]))).cache()
                
                
                self._M = y_len.first()
                
                        
                    
                if len(y_len.distinct().collect())!=1:
                    print 'All label vectors should have the same length. Aborting.'
                    return False
                self._data = data.map(lambda x: (Vectors.dense(x[0]),
                                                 Vectors.dense(x[1])))
                return True
            except:
                return False
    
        elif self._typeMVA == 'PCA':
            #Case 3: If MVA is PCA, then RDD elements should be numpy arrays
            try:
                x_len = data.map(lambda x: len(Vectors.dense(x))).cache()
                self._numVariables = x_len.first()
                
               
                self._M = self._numVariables
                
                 
                
                if len(x_len.distinct().collect())!=1:
                    print 'All feature vectors should have the same length. Aborting.'
                    return False
                self._data = data.map(lambda x: (Vectors.dense(x),
                                                 Vectors.dense(x)))
                return True
            except:
                return False

        return False


    def calcCov(self,typeCov):
        """
        This function calculates the covariance matrix for the training data

        :param typeCov:
            Type of covariance matrix to be calculated, it can be Cyx or Cyy
        """

        if typeCov == 'Cyx' :
            Cyx = self._data.map(lambda x : np.dot(x[0][:,np.newaxis],x[1][:,np.newaxis].T)).mean()
            Cov=Cyx
        elif typeCov == 'Cyy':
            Cyy = self._data.map(lambda x : np.dot(x[0][:,np.newaxis],x[0][:,np.newaxis].T)).mean()
            Cov=Cyy
        else:
            print 'This type of covariance matrix cannot be calculated'
        return Cov


    def createOmega(self):
        """
        This function creates the Omega matrix for the step U and step W,
        it depends of the type of MVA method.

        """
        if self._typeMVA in ["PCA", "OPLS"] :
            Omega = np.eye(self._M)

        else :
            Cyy = self.calcCov('Cyy')
            Omega=np.linalg.inv(Cyy)

        return Omega

        
    def calcFrobeniusNorm(self,Uold,Unew):
        """
        This function calculate the Frobenius norm between two matrices

        """
        A=Uold-Unew
        return lin.norm(A,'fro')    


    def normalizer(self):
        """
        This function normalize the training data
  
        """
  
        if self._typeNorm == 'norm':
            #Normalize input features
            RDD_X = self._data.map(lambda x: x[1])
            self._scaler = StandardScaler(withMean=True, withStd=True).fit(RDD_X)
            RDD_X_norm = self._scaler.transform(RDD_X)
            RDD_Y = self._data.map(lambda x: x[0])
            RDD_Y_norm = StandardScaler(withMean=True, withStd=False).fit(RDD_Y).transform(RDD_Y)
        else:
            #Normalize input features
            RDD_X = self._data.map(lambda x: x[1])
            self._scaler = StandardScaler(withMean=True, withStd=False).fit(RDD_X)
            RDD_X_norm = self._scaler.transform(RDD_X)
            if self._typeMVA == 'PCA':
                RDD_Y = self._data.map(lambda x: x[0])
                RDD_Y_norm = StandardScaler(withMean=True, withStd=False).fit(RDD_Y).transform(RDD_Y)
            else:
                RDD_Y_norm = self._data.map(lambda x: x[0])

        # Create a new RDD of LabeledPoint data using the normalized features
        self._normdata = RDD_Y_norm.zip(RDD_X_norm)


    def stepU(self,W,Omega, R):
        """
        This function calculate the step U 
  
        :param W:
            W matrix
        :param Omega:
            Omega matrix
        :param R:
            Number of distinct classes minus one
        
        """
        U = np.empty((R,self._numVariables))

        for r in range(R):
            print 'Extracting projection vector ' + str(r) + ' out of ' + str(len(range(R)))
            Wr = W[:,r][:,np.newaxis]
            def createPseudoY(Y, W, Omega):
                """
                This function calculates Y' = W^TOmegaY for the step U
  
                :param Y:
                    RDD of labels or outputs
                :param W:
                    W matrix calcutated in step W
                :param Omega:
                    Omega matrix 
                """  
                return np.squeeze(W.T.dot(Omega).dot(Y.T))
            PseudoY = self._normdata.map (lambda x : createPseudoY(x[0], Wr, Omega))
            Datar = self._normdata.zip(PseudoY).map(lambda x: LabeledPoint(x[1], x[0][1]))
            # Build the model
            lr = LinearRegressionWithSGD.train(Datar, iterations=self._iterations, regType=self._typeReg, regParam=self._regParam, step=self._step)
            U[r,:] = lr.weights

        return U


    def stepW(self, U, Cyx, Omega, Omega_1):
        """
        This function calculates the step W
  
        :param U:
            U matrix calculated in step U
        :param Cyx:
            The covariance matrix between the labels or outputs and the features
        :param Omega:
            Omega matrix
        :param Omega_1:
            The inverse of the omega matrix
        """
        print U.shape
        print Cyx.shape
        print Omega.shape
        A = Omega.dot(Cyx).dot(U.T)
        V, D, V2 = np.linalg.svd(A,full_matrices=False)
        W = np.dot(Omega_1,V)
        
        return W


    def computeMSE(self, U, W, trainingData):
        """
        This function compute de MSE

        :param U:
            U matrix 
        :param W:
            W matrix
        :param trainingData:
            RDD of training data
        """
        return trainingData.map(lambda x: np.mean(np.array(x.codedLabel - np.dot(W,np.dot(x.features, U.T)))**2)).mean()


    def fit(self, data):
        """
        This function fits the model. It calculates de matrix U where each 
        column is a vector containing the coefficients for each extracted feature.

        :param data:
            
        """

        if self.prepareData(data):
            
            Omega= self.createOmega()
            Omega_1=np.linalg.inv(Omega)
            num_Ustep=0
            
            #Normalize data
            self.normalizer()

            #Initialize U and W variables
           
            R = int(np.minimum(self._M-1, self._R))
            U_old = np.empty((R,self._numVariables))
            Cyx=self.calcCov('Cyx')
            W = self.stepW(U_old,Cyx,Omega,Omega_1)
            U_new = self.stepU(W,Omega,R)
            
            
            while (self.calcFrobeniusNorm(U_old,U_new) > self._tol) and (num_Ustep<self._max_Ustep) :
                U_old=U_new
                W = self.stepW(U_old,Cyx,Omega,Omega_1)
                U_new = self.stepU(W,Omega,R)
                num_Ustep=num_Ustep + 1
                
                if num_Ustep==self._max_Ustep :
                    print 'You have reach the max number of U step, change the tolerance'
                    
                print 'Frobenius norm error: ' + str(self.calcFrobeniusNorm(U_old,U_new))

            self._U=U_new
            self._W=W
    
    def predict(self, RDD_X2):
        """
        This function find relevant features by combining X=U^T*X2. It is
        needed to fit the model first

        :param sc: SparkContext
        :param RDD_X2:
          Training data 
        """
        if self._U != None:
            RDD_norm = self._scaler.transform(RDD_X2)
            U = self._U
            RDD=RDD_norm.map(lambda x: x.dot(U.T))
            return RDD  
        else :
            print 'You have to fit the model first'

