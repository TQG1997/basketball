import numpy as np

X = np.load(r"C:\Users\EDC\Desktop\output.npy")
Y = X[~np.isnan(X)]
np.savetxt('E://out.csv', [Y], delimiter=',')
