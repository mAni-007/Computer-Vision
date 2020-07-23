import torch
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 
from sklearn import datasets 

n_pts = 500
x, y = datasets.make_circles(n_samples = n_pts, random_state = 123, noise = 0.1, factor = 0.2)
x_data = torch.Tensor(x)
print (x[y == 0,0])
# print (len(x), '\n',len(y))
y_data = torch.Tensor(y.reshape(500,1 ))
# print ("x_data: ", x_data, "y_data: ", y_data)

def scatter_plot():
	plt.scatter(x[y==0,0], x[y == 0,1])
	plt.scatter(x[y==1,0], x[y == 1,1])
	plt.show()


class Model(nn.Module):
	def __init__ (self, input_size, H1, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, H1)
		self.linear2 = nn.Linear(H1, output_size)

	def forward(self, x):
		x = torch.sigmoid(self.linear(x))
		x = torch.sigmoid(self.linear2(x))
		return x
	def predict(self, x):
		pred = self.forward(x)
		if pred >= 0.5:
			return 1
		else :
			return 0


torch.manual_seed(2)
model = Model(2, 4, 1)
# print(list(model.parameters()))


#Time to train 
criteroin = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

#Training time
epochs = 1000
losses = []
for i in range(epochs):
	y_pred = model.forward(x_data)
	loss = criteroin(y_pred, y_data)
	# print("epoch: ", i, "loss", loss.item())
	losses.append(loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# # plt.show()

#buliding thwe upraded graph using the unpadted weights
def plot_decision_boundary(x, y):
	x_span = np.linspace(min(x[:, 0]) -0.25, max(x[:, 0]) + 0.25)
	y_span = np.linspace(min(x[:, 1]) -0.25, max(x[:, 1]) + 0.25)
	xx, yy = np.meshgrid(x_span, y_span)
	grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
	pred_func = model.forward(grid)
	z = pred_func.view(xx.shape).detach().numpy()
	plt.contourf(xx,yy, z)


plot_decision_boundary(x, y)
scatter_plot()




#Test cases from here onward
x = 0.025
y = 0.025
point = torch.Tensor([x, y])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("Prediction is", prediction)
plot_decision_boundary(x, y)