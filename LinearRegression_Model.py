import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np 


#Creating the dataset 
X = torch.randn(100,1)*10
y = X + 3*torch.randn(100,1)
plt.plot(X.numpy(), y.numpy(), 'o')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

 
#Boilerplate CODE------intializing instances 
class LinearReg(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	def forward(self, x):
		pred = self.linear(x)
		return pred

#Checking the class
#Seed is for to provide random value to weight and bias
torch.manual_seed(1)
model = LinearReg(1,1)

#defining/extracting the parameters from model
[w, b] = model.parameters()
print (w, b)
def get_params():
	return (w[0][0].item(), b[0].item())

#creating the line for w1 and b1 
def plt_fit(title):
	plt.title = title
	w1, b1 = get_params()
	x1 = np.array([-30, 30])
	y1 = w1*x1 + b1
	y2 = 0.9981 *x1 -0.0253
	plt.plot(x1, y2, 'b')
	plt.plot(x1, y1, 'r')
	plt.scatter(X,y)
	plt.show()
plt_fit('Initial Model')


#differention and MSE define here, now to use them in loop which will be in for loop
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


#Training starts from here
epochs = 100
losses = []
for i in range(epochs):
	y_pred = model.forward(X)
	loss = criterion(y_pred, y)
	# print( "epoch: ", i, "loss: ", loss.item())
	
	losses.append(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt_fit('Trained Model')
print( w, b)