# Streamlit App
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


st.title('Gradient Descent Algorithm for Linear Regression')

txt = st.write("The gradient descent algorithm for linear regression involves" 
               "updating the parameters iteratively to minimize the" "cost function. The cost function for linear regression is often the mean squared error (MSE).")

#st.header('The weight update happens as follows', divider='rainbow')
#st.header('_Streamlit_ is :blue[cool] :sunglasses:')

st.latex(r'''\theta_j^{k+1} = \theta_j^k - \alpha \cdot \frac{\partial J}{\partial \theta_j^k}''')

#st.divider()
# Sidebar
st.sidebar.header('Parameters')
learning_rate = st.sidebar.slider('Learning Rate',min_value=3e-4, max_value=0.1, value=0.001)
epochs = st.sidebar.slider('Epochs', min_value=100, max_value=1000, value=500)


# Generate random data
df=pd.read_csv("california_housing_train.csv")
X=np.array(df["median_income"]).reshape(-1,1)
y=np.array(df["median_house_value"]).reshape(-1,1)

# Create model and fit
from gradient_descent import GradDescent
model=GradDescent()
theta, loss = model.fit(X, y, learning_rate=learning_rate, epochs=epochs)
ypred = model.predict(X)

MAE=np.mean(abs(ypred-y))
MSE=np.mean((ypred-y)**2)
RMSE=np.sqrt(MSE)
RSS=np.sum((ypred-y)**2)

col1,col2= st.columns(2)
col1.metric("Residual Sum of Squares",round(RSS,3))
col2.metric("Mean Absolute Error",round(MAE,3))


# Plotting with reduced size
fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figsize parameter to your preference
ax.scatter(X[:150, :], y[:150, :], color='blue')
ax.plot(X[:150, :], ypred[:150, :], color='red')
ax.set_xlabel('Median Income (in USD)')
ax.set_ylabel('Median House Value (in USD)')
st.pyplot(fig)