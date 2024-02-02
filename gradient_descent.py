import numpy as np 
import matplotlib.pyplot as plt

class GradDescent:
      def __init__(self,theta=None,loss=None):
          self.theta=None
          self.loss=loss

      def fit(self,X,y,learning_rate,epochs):
          #expected_shape = 1
          #assert X.shape[1] == expected_shape, f"Shape mismatch: Expected {expected_shape}, but got {X.shape[1]}"
          
          X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
          self.theta=np.random.randn(X.shape[1],1)    
          self.loss=[]
          for i in range(epochs):
            ypred=np.matmul(X,self.theta)
          
            self.loss.append(np.mean((ypred-y)**2))
            grad=(2/(X.shape[0]))*(np.matmul(X.T,(ypred-y)))

            self.theta=self.theta-learning_rate*grad
         
          return np.array(self.theta), np.array(self.loss) 

      def predict(self,X):
          X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
          return np.matmul(X,self.theta)

#X=np.random.randn(1000,1)
#y=3+2*X+np.random.randn(1000,1)

#from sklearn.datasets import make_regression
#X,y=make_regression(n_samples=1000,n_features=5,n_informative=5,n_targets=1)
#y=y.reshape((y.shape[0],1))

#model=GradDescent()
#theta,loss=model.fit(X,y,learning_rate=0.001,epochs=500)
#print(theta.shape)
#ypred=model.predict(X)
#print(ypred.shape)
#plt.scatter(X[:,0],y,color='blue')
#plt.plot(X[:,0],ypred,color='red')
#plt.show()
