import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse

colors = [
	'#d62728',  # Brick Red
	'#ff7f0e',  # Safety Orange
	'#bcbd22',  # Tumeric Yellow-Green
	'#2ca02c',  # Cooked Asparagus Green
	'#17becf',  # Blue-Muted Cyan
	'#1f77b4',  # Muted Blue
	'#9467bd',  # Muted Purple
	'#e377c2',  # Raspberry Pink
	'#8c564b',  # Chestnut Brown
	'#7f7f7f'   # Middle Gray
] # gemini came up with this

def runSweep(N, epsilon, device):
	cnt = np.zeros((9, N))
	kwargs = {'device':device, 'dtype':torch.float16}
	for p in range(5, 14):
		dim = 2**p
		i = 0
		db = torch.zeros(1, dim, **kwargs)
		batch_size = 1
		while i < N:
			if p > 10:
				if i >= 64:
					batch_size = 2
				if i >= 256:
					batch_size = 8 # speed things up slightly
			cand = torch.randn(batch_size, dim, **kwargs)
			cand = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-12)
			dp = db @ cand.T # with batched, this will be a matrix
			if torch.max(torch.abs(dp)) < epsilon: # so this will be reject all
				db = torch.cat((db, cand), dim=0)
				n = db.shape[0]-1
				print(f"{dim} cnt[{i}] = {n}")
			n = db.shape[0]-1
			cnt[p-5,i:i+batch_size] = n
			i = i+batch_size
		# save after each pass
		np.save('concentration_cnt.npy', cnt)
	return cnt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="demo of the concentration of measure")
	parser.add_argument('-n', type=int, default=100000, help="How many vectors to test",)
	cmd_args = parser.parse_args()
	N = cmd_args.n - (cmd_args.n % 8) # avoid batch size edge cases

	device = torch.device("cuda")
	try:
		cnt = np.load('concentration_cnt.npy')
	except FileNotFoundError :
		print("Saved results not found, generating..")
		cnt = runSweep(N, 0.1, device)
		np.save('concentration_cnt.npy', cnt)
	for p in range(5, 14):
		plt.plot(cnt[p-5, :], color=colors[p-5], label=f"{2**p}")
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
