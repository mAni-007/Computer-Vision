import torch
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
from sklearn import datasets


#creating the datasets
npoints =  100
centers = [[-0.5, 0.5], [0.5, -0.5]]

x,y = datasets.make_blobs(n_samples = npoints, random_state = 123, centers = centers, cluster_std = 0.4)
x_data = torch.Tensor(x)
y_data = torch.Tensor(y.reshape(100,1))
# print(y)
# print(x[y == 1,0])
# print(x_data[y == 1,1])
# print(x_data)

def scatter_plot():
	plt.scatter(x[y==0,0], x[y==0,1])
	plt.scatter(x[y==1,0], x[y==1,1])
	# plt.show(100)

scatter_plot()


# Model initialization
class Model(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	def forward(self, x):
		pred = torch.sigmoid(self.linear(x))
		return pred
	# def predict(self,  x):
	# 	pred = self.forward(x)

torch.manual_seed(2)
model = Model(2,1)
# print(list(model.parameters()))

#printing the parameters
[w,b] = model.parameters()
w1, w2 = w.view(2)
def get_params():
	print (w1.item(), w2.item(), b[0].item())
	return (w1.item(), w2.item(), b[0].item())
get_params()


#Drawing the line on the graph using the initial paramters
def plot_fit(title):
	plt.title = title
	w1, w2, b1 = get_params()
	x1 = np.array([-2,2])
	x2 = (w1*x1 + b1)/(-w2)
	plt.plot(x1,x2, 'r')
	scatter_plot()
	plt.show()
# plot_fit('initial Model')


#Time to training 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

epochs = 100
losses = []
for i in range(epochs):
	y_pred = model.forward(x_data)
	loss = criterion(y_pred, y_data)
	print("epochs: ", i, "loss: ", loss.item())
	losses.append(loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epochs')
# plt.grid()
plot_fit('Trained model')








