import numpy as np

total_num = 69219
n_nets = 5
alpha = 0.5
n_round = 10

idxs = np.random.permutation(total_num)
overall_batch_idxs = np.array_split(idxs, n_nets)
idx_batch_temp = []
idx_batch = [[0]*10]*5
# net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
for i in range(n_nets): 
    proportions = np.random.dirichlet(np.repeat(alpha, n_round))
    proportions = (np.cumsum(proportions) * len(overall_batch_idxs[i])).astype(int)[:-1]
    idx_batch_temp.append(np.split(overall_batch_idxs[i], proportions))
    idx_batch[i][0] = idx_batch_temp[i][0].tolist()

    for j in range(1, n_round):
        prior = idx_batch[i][j-1]
        present = idx_batch_temp[i][j]
        idx_batch[i][j] = prior + present.tolist()
    
for i in range(10):
    print(len(idx_batch[0][i]))