import pandas as pd
import numpy as np
from sklearn import tree
arbol = tree.DecisionTreeClassifier()
df = pd.read_csv('Datos.csv')
df = df.sample(frac=1.0, random_state=1);
columnas = df.columns
P = np.array(df[columnas[1:17]])
T = np.array(df[columnas[17]])
entrada = int(len(P)*0.7)
inP = P[:entrada]
inT = T[:entrada]
test_P = P[entrada:len(P)]
test_T = T[entrada:len(T)]
arbol.fit(inP,inT)
print(arbol.score(P,T))
tree.export_graphviz(arbol)
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(arbol, out_file=f,filled=True)
