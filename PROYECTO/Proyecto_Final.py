import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

class RegresionMultiple():
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.fil, self.col = x.shape

  #### CREAMOS LA MATRIZ DE DATOS X ####
  def Matriz_X(self):
    fil, col = self.fil, self.col
    X = np.ones((fil, col+1))  # Se crea una matriz de 1 con dimensiones : numero de filas y numero de columnas con los datos del DataFrame 'x'
    X[:,1:] = self.x  #Colocamos los datos desde la segunda columna, hasta la ultima, La primera columna es de solo  
    return X
  
  ### CREAMOS BETA DE LA FORMULA CON MINIMOS CUADRADOS ###
  def Minimos_Cuadrados(self):
    X = self.Matriz_X()
    X_tran = np.transpose(X)
    beta_hat = np.linalg.inv(X_tran @ X) @ X_tran @ self.y
    return beta_hat

  ### PREDICCION DE LOS DATOS 'y' NUEVOS ###
  def Prediccion_y_(self):
    X = self.Matriz_X()
    beta_hat = self.Minimos_Cuadrados()
    y_predict = X @ beta_hat
    return y_predict
  
  ### ERROR CUADRATICO MEDIO ###
  def ErrorCuadraticoMedio(self):
    y_predict = self.Prediccion_y_()
    mse = np.mean((y_predict - self.y) ** 2)
    return mse
  
  # FUNCION PARA VALIDACION CRUZADA --- 'y' predecida #
  def Evaluar(self,x):
    if x.ndim == 1:
      x = x.reshape(-1, 1)
    A = np.hstack((np.ones((x.shape[0], 1)), x))
    coeficientes = self.Minimos_Cuadrados()
    return np.dot(A, coeficientes)
  
  ### COEFICIENTE DE DETERMINACION R2 ###
  def R2(self, y_hat, y, P = None):
    n = len(self.y)
    y_bar = np.mean(self.y) # Calcular la media de los valores reales

    if P == None:

      # Calcular la suma total de los cuadrados de las diferencias entre los valores reales y la media
      R2 = sum( (y_hat - y_bar)**2 ) / sum( (y - y_bar)**2 )
      #print(R2)
      return R2

    # R2 AJUSTADO
    else:
      num=sum((y_hat-y)**2)
      dem=sum((y-y_bar)**2)
      R2_ajustado = 1-((n-1)/(n-P))*(num/dem)
      return R2_ajustado
  
  ### VALIDACION CRUZADA ###
  def validacion_cruzada2(self, k, P='N'):
    indices = np.random.permutation(self.fil)  # Se realiza una permutación aleatoria de los índices de los datos
    particiones = [indices[j::k] for j in range(k)]  # Se dividen los índices en k particiones
    resultados = []

    for i in range(k):
      prueba = particiones[i]
      entrenamiento = np.concatenate([particiones[j] for j in range(k) if j != i])  # Se concatenan los índices de las particiones de entrenamiento

      x_entrenamiento, y_entrenamiento = self.x[entrenamiento], self.y[entrenamiento]
      x_prueba, y_prueba = self.x[prueba], self.y[prueba]
      modelo_i = RegresionMultiple(x_entrenamiento, y_entrenamiento)  # Se crea un nuevo modelo para cada partición de entrenamiento
      y_hat_prueba = modelo_i.Evaluar(x_prueba)  # Se obtienen las predicciones en los datos de prueba

      if str(P) == 'N':
        R2 = modelo_i.R2(y_hat_prueba, y_prueba)  # Coeficiente de determinación estándar
      else:
        R2 = modelo_i.R2(y_hat_prueba, y_prueba, P)  # Coeficiente de determinación ajustado

      resultados.append(R2)

    R2_promedio = np.mean(resultados)  # Promedio de los coeficientes de determinación obtenidos en cada partición
    return R2_promedio

  ### GRAFICA CON LAS 'y' extraidas de los datos y las 'y_predict' que son las que se predicen con el modelo de regresion lineal multiple ###
  def Grafica(self):
    y_predict = self.Prediccion_y_()

    plt.figure(figsize=(5,5))
    plt.scatter(self.y, y_predict)
    plt.xlabel('Precios originales')
    plt.ylabel('Precios predecidos')
    plt.title('Comparacion de los precios reales y predecidos')
    plt.show()

### ANALISIS DE COMPONENTES PRINCIPALES (PCA) ###

class PCA(RegresionMultiple):
  num_comp=0
  def __init__(self, X, Y, margen):
    self.X = X
    self.Y = Y
    self.filas, self.columnas = X.shape

    self.mediaX = X.mean(axis=0)
    self.DesvStanX = X.std(axis=0)

    self.mediaY = Y.mean(axis=0)
    self.DesvStanY = Y.std(axis=0)

    X_estandarizado = self.Estandarizar(self.X)

    self.matrizPro = self.MatrizProyeccion(margen, X_estandarizado)
    Z = self.Proyectar(X_estandarizado)

    super().__init__(Z, Y)
    
  def Estandarizar(self, X):
    media = self.X.mean(axis = 0)
    desvStand = self.X.std(axis = 0)
    X_escalado = np.zeros_like(X)

    for k in range(self.columnas):
      if desvStand[k] != 0:
        X_escalado[:,k] = (X[:,k] - media[k]) / desvStand[k]
      else:
        X_escalado[:,k] = X_escalado[:,k]
    return X_escalado
  
  def MatrizProyeccion(self, margen, X):
    U, S, VT = np.linalg.svd(X) # Descomposición en valores singulares
    V = VT.T # Transponemos la matriz V para que las columnas sean los vectores propios de X.T@X
    S = S**2 # Obtener los valores propios = cuadrado de los valores singulares

    # Calcular el número de componentes principales
    porcVarAcum = 0
    k = 1
    while porcVarAcum < margen:
      porcVarAcum = sum(S[:k]) / sum(S)
      k += 1
    PCA.num_comp = k

    matrizPro = V[:,:PCA.num_comp]#La matriz de proyección
    print("Número de componentes principales utilizadas:", PCA.num_comp)

    return matrizPro
  
  def R2(self,y_hat,y):
    R2 = super().R2(y_hat,y,PCA.num_comp)
    return R2

  def Validacion_Cruzada_pca(self, k):
    return super().validacion_cruzada2(k)
  
  def Proyectar(self,Datos):
    Z = Datos @ self.matrizPro
    return Z
  
  def Predecir(self,Datos):
    Datos_estandarizados = self.Estandarizar(Datos)
    Datos_proyectados = self.Proyectar(Datos_estandarizados)
    y_hat = super().Evaluar(Datos_proyectados)
    return y_hat

### LECTURA DE DATOS DEL CSV
boston = pd.read_csv("boston.csv")
datos = pd.DataFrame(boston)
datos.dropna(inplace = True) # LIMPIAMOS LOS DATOS CON VALOR 'NaN'

#### DEFINIMOS LAS VARIABLES X and Y ####
# Usando 'drop' de pandas, seleccionamos del dataframes todos los datos y eliminamos el que no necesitemos. Con axis = 1 para eliminar la columna
x = datos.drop(['MEDV'], axis = 1).values
y = datos['MEDV'].values

### DIVISION DE DATOS. ENTRENAMIENTO/PRUEBA ###
# RELACION 80/20 #
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x, y, test_size=0.20, random_state=42)

print(boston.head())


### INVOCAR A LA CLASE REGRESION LINEAL MULTIPLE ###
Regresion = RegresionMultiple(x_prueba,y_prueba)
y_predict = Regresion.Prediccion_y_()

valid = Regresion.validacion_cruzada2(5)
print('Validacion cruzada con 5:', round(valid,4))

R2 = Regresion.R2(y_predict, y_prueba)
print('\nCoeficiente de determinacion R2:', round(R2,4))

R2_2 = r2_score(y_prueba , y_predict)
print('Coeficiente R2 sklearn: ', round(R2_2,4))

MSE = Regresion.ErrorCuadraticoMedio()
print('\nError cuadratico medio:', round(MSE,4))

mse = mse=mean_squared_error(y_prueba ,y_predict)
print('Error cuadratico sklearn:', round(mse,4))

# Mostrar grafica
Regresion.Grafica()

### COMPARACION DE DATOS ORIGINALES Y PREDICHOS EN DATAFRAME ###
datos_y_pred = pd.DataFrame({'Precios reales': y_prueba, 'Precios Predecidos': y_predict, 'Diferencia': y_prueba - y_predict})
datos_y_pred[0:500]
print(datos_y_pred)

### INVOCAR CLASE PCA ###

Componentes_principales = PCA(x, y, 0.6)
y_predict_pca = Componentes_principales.Prediccion_y_()

Validacion_Cruzada_pca = Componentes_principales.Validacion_Cruzada_pca(5)
print('\nValidacion cruzada con PCA: ', Validacion_Cruzada_pca)

R2_ajustado = Componentes_principales.R2(y_predict_pca,y)
print('\nCoeficiente de determinacion R2 ajustado: ', R2_ajustado)

mse_pca = Componentes_principales.ErrorCuadraticoMedio()
print('\nError cuadratico medio: ', mse_pca)

Componentes_principales.Grafica()