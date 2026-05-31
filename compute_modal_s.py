import numpy as np
Phi = np.array([[0.37024474, -0.97779098, 3.10754624],[0.65544238,-0.78924412,-3.86619826],[1.0,1.0,1.0]])
Gamma = np.array([1.33287597,-0.37718726,0.04431129])
M = np.diag([20.0,15.0,15.0])
# compute s_n = Gamma_n * (M @ phi_n)
s = np.column_stack([Gamma[i] * (M @ Phi[:, i]) for i in range(3)])
np.set_printoptions(precision=6, suppress=True)
print('s (each column is mode):')
print(s)
print('\nsums (M_n^* from sum s_in):')
print(np.sum(s, axis=0))
# Also compute M_star via L_n^2 / M_n using values from data
L_n = np.array([32.23653050956122, -16.39448137202328, 19.157950862462059])
M_n = np.array([24.18569407279455, 43.465098176420625, 432.3492077507844])
M_star = L_n**2 / M_n
print('\nM_star (L_n^2 / M_n):')
print(M_star)
