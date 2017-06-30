import torch.nn as nn


class Autoencoder(nn.Module):
	def __init__(self, in_dim=784, h_dim=400):
		super(Autoencoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(in_dim, h_dim),
			nn.ReLU()
			)

		self.decoder = nn.Sequential(
			nn.Linear(h_dim, in_dim),
			nn.Sigmoid()
			)


	def forward(self, x):
		"""
		Note: image dimension conversion will be handled by external methods
		"""
		out = self.encoder(x)
		out = self.decoder(out)
		return out
