import wooldridge as woo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargamos el dataset CRIME1
data = woo.data('crime1')


# Selección de variables independientes y dependiente
X_raw = data[['pcnv', 'tottime', 'ptime86', 'inc86', 'qemp86']].values
y = data['narr86'].values

# Estandarización de las variables independientes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)  # Escalamos únicamente las variables independientes

# Agregamos el intercepto (columna de 1s) a las variables independientes estandarizadas
X_standardized = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Cálculo de la matriz X'X
XTX = np.dot(X_standardized.T, X_standardized)

# Cálculo del vector X'y
XTy =  np.dot(X_standardized.T, y)

# Resolución del sistema de ecuaciones X'X * beta = X'y
# Utilizamos la función de NumPy para resolver sistemas lineales
beta_standardized = np.linalg.solve(XTX, XTy)


# Mostrar los coeficientes calculados con las variables estandarizadas
print("Coeficientes del modelo con variables estandarizadas (beta):")
print(beta_standardized)

# Interpretación de los coeficientes
# beta[0] -> Intercepto
# beta[1] -> Coeficiente para pcnv
# beta[2] -> Coeficiente para tottime
# beta[3] -> Coeficiente para ptime86
# beta[4] -> Coeficiente para inc86
# beta[5] -> Coeficiente para qemp86
