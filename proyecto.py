#-------Seccion de librerias BEGIN.-----#
import imp
import warnings 
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from pyparsing import col
import seaborn as sns
from sklearn import preprocessing
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import pingouin as pg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#-------Seccion de librerias END.-----#

#-------Configuracion de la consola(Salidas de codigo) BEGIN.-----#
warnings.filterwarnings("ignore") # ignorar alertas
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',15)
pd.set_option('display.width',1000)
#-------Configuracion de la consola(Salidas de codigo) END-----#

#-------Lectura e inspeccion de datos BEGIN-----#
df = pd.read_csv("Car_Prices_Poland_Kaggle.csv") #Importacion de datos
print(df.info())                                 #Obtenemos los tipos de datos decada columna
print(df.describe())                             #Obtenemos valores estadisticos de las columna que tengan valores numericos
print(df.head())                                 #Obtenemos los primeros 5 registros
print(df.tail())                                 #Obtenemos los ultimos 10 rehistros
print(df.sample(10))                             #Obtenemos 10 registros de ejemplo
df.rename(columns={"Unnamed: 0": "ID"},          #Cambiamos el nombre de la columna "Unnamed: 0" por "ID"
               inplace=True)
#-------Configuracion de la consola(Salidas de codigo) END-----#

#-------Pre procesamiento BEGIN-----#
print(df.isnull().sum())                        #Deteccion de datos nulos
print(df['generation_name'].sample(10))         #Ejemplo de la columna con datos nulos
del df["generation_name"]                       #Eliminacion de la columna que contiene datos nulos
print(df.isnull().sum())                        #Verificamos que se elimino la columna
print(df.describe())                            #Localizacion de datos atipicos mediante maaximos y minimos en la columnas numericas
df.drop(['ID'],1).hist()                        #Histogramas sin la columna de ID
plt.show()                                      #Lanzamiento del histograma

#-------Eliminacion de Outliers – Inliers BEGIN-----#
columnas_dataSet = ['price', 'mileage', 'year', 'vol_engine']               #Defininimos un arreglo con el nombre de columnas con outliers
for i in columnas_dataSet:                                                  #For para eliminacion de datos atipicos
    plt.title(i)                                                            #Titulo de la grafica
    plt.boxplot(df[i], vert=False)                                          #Carga de la columna en la grafica de caja y bigote
    plt.show()                                                              #Lanzamiento de la grafica
    q1 = df[i].quantile(0.25)                                               #Otencion del primer cuartil
    q3 = df[i].quantile(0.75)                                               #Obtencion del tercer cuartil
    IQR = q3 - q1                                                           #Obtencion del indice IQR
    bi_calculado = (q1 - 1.5 * IQR)                                         #Obtencion del limite inicial
    bs_calculado = (q1 + 1.5 * IQR)                                         #Obtencion del limite final
    ubicacion_outliers = (df[i] >= bi_calculado) & (df[i] <= bs_calculado)  #Obtencion de los datos NO atipicos
    df = df[ubicacion_outliers]                                             #Guardado del data set sin estos dato
    plt.title(i)                                                            #Titulo de la grafica 
    plt.boxplot(df[i], vert=False)                                          #Carga de la columna en la grafica de caja y bigote
    plt.show()                                                              #Lanzamiento de la grafica


df.reset_index(inplace=True, drop=True)                                     #Reseteo de indices predeterminado
df['ID'] = np.arange(0,len(df))                                             #Reseteo de nuestra columna de indices

df.drop(['ID'],1).hist()                                                    #Histogramas sin la columna de ID
plt.show()                                                                  #Lanzamiento del histograma
print(df.shape)                                                             #Obtenemos la cantidad de datos resultantes despues de la limpia

#-------Eliminacion de Outliers – Inliers END-----#


#-------Creacion de etiquetas Begin----#
encoder = preprocessing.LabelEncoder()                                          #Cargamos el codificador
encoder.fit(df["mark"])                                                         #Pasamos la columna de donde tomara las etiquetas
aux_mark = encoder.transform(df["mark"])                                        #Tranformamos la columna
dfMark_transform = pd.DataFrame(aux_mark)                                       #Transformamos el resultado en un data set
df['mark_transform'] = dfMark_transform                                         #Añadiamos la columna a nuestro data set original

clasesmark = pd.DataFrame(list(encoder.classes_), columns=['Mark_Labels'])      #Creamos un data set con las equivalente 
aux_mark_etiquetado = encoder.transform(clasesmark["Mark_Labels"])              #Guardamos los equivalentes
clasesmark['mark_transform'] = aux_mark_etiquetado                              #Lo añadimos al dataset de equivalencias

encoder = preprocessing.LabelEncoder()                                          #Cargamos el codificador
encoder.fit(df["fuel"])                                                         #Pasamos la columna de donde tomara las etiquetas
aux_mark = encoder.transform(df["fuel"])                                        #Tranformamos la columna
dfMark_transform = pd.DataFrame(aux_mark)                                       #Transformamos el resultado en un data set
df['fuel_transform'] = dfMark_transform                                         #Añadiamos la columna a nuestro data set original

clasesfuel = pd.DataFrame(list(encoder.classes_), columns=['Fuel_Labels'])      #Creamos un data set con las equivalente 
aux_fuel_etiquetado = encoder.transform(clasesfuel["Fuel_Labels"])              #Guardamos los equivalentes
clasesfuel['Fuel_transform'] = aux_fuel_etiquetado                              #Lo añadimos al dataset de equivalencias

print(clasesmark)                                                               #Columna mark transformada
print(clasesfuel)                                                               #Columna fiel transformada
print(df.head(10))                                                              #Verificamos datos iniciales
print(df.tail(10))                                                              #Verificamos datos finales
print(df.isnull().sum())                                                        #Verificamos nulos
#-------Creacion de etiquetas END-----#

#-------Categorizacion de datos BEGIN-----#


df['precios_cat'] = pd.cut(x=df['price'],
                     bins=[np.NINF, 38300,76000, np.inf],
                     labels=["economico","medio","lujo"])
df['mileage_cat'] = pd.cut(x=df['mileage'],
                     bins=[np.NINF, 67845, 135750,203625, np.inf],
                     labels=["Agencia","Rural","Urbano","Viaje"])
print(df[['price','precios_cat','mileage','mileage_cat']].sample(20))
#-------Categorizacion de datos BEGIN-----#

#-------Normalizacion Begin -----#

scale= StandardScaler()                                                         #cargamos el metodo para estandarizar
xnorm= df[['mileage','vol_engine','price']]                                     #aasignamos las columnas a normalizar
scaledx = scale.fit_transform(xnorm)                                            #Aplicamos la normalizacion    
auxnorm = pd.DataFrame(scaledx, columns=['mileage_norm','vol_engine_norm','price_norm'])                        #Guardamos en data set auxiliar
df[['mileage_norm','vol_engine_norm','price_norm']] = auxnorm[['mileage_norm','vol_engine_norm','price_norm']]  #Guardamos en data set original
print(df.isnull().sum())                                                                                        #Verificamos que se haya insertado bien
#-------Normalizacion END-----# 

#-------Pre procesamiento END-----#

#---------Interprestacion de datos BEGIN------#

pastel = df.fuel.value_counts()                             #Contabilizamos los datos de la columna fuel
explode = (0.1, 0, 0, 0, 0.2, 0.3)                          #Defininimos cual porcentaje estara separado
pastel.plot.pie(autopct = '%1.1f%%', explode = explode,)    #cargamos los dato
plt.axis("equal")
plt.show()

print(df.groupby(by='year')['price'].mean())                                    #En consola imprimimos la media de los precios por año
print(pd.crosstab(df['year'], df['fuel'], values=df['price'], aggfunc='mean'))  #En consola imprimimos la media del precio segun el año y el tipo de combustible

pd.crosstab(df['year'], df['fuel']).plot()      #Grafica de cantidas de autos por tipo de combustible por añ
plt.show()                                      #Lanzamos grafica

pd.crosstab(df['year'], df['fuel'], values=df['price'], aggfunc='mean').plot()  #Grafica de la media de precios segun el combustible a lo largo de los años
plt.show()                                                                      #Lanzamos la grafica

df.groupby(by='year')['price'].mean().plot()    #Grafica de la media de los precios a lo largo de los años
plt.show()                                      #Lanzamos la grafica

sns.countplot(x="mark", data=df, order=df["mark"].value_counts().index)         #Cargamos los datos
plt.xticks(rotation=60)                                                         #La etiqueta de la columnas la rotamos
plt.xlabel("Mark", size=15)                                                     #Lo que mostrara el eje x
plt.ylabel("Count", size=15)                                                    #Lo que mostrara el eje y
plt.title("Cantidad por MARK", size=15)                                         #Lo que mostrara el titulo
plt.show()                                                                      #Lazamos la grafica


car_corr = df.corr(method='spearman')           #defiimos el metodo por el cual se hara la matriz de correlacion 
sns.heatmap(car_corr,                           #Cargamos los datos
              xticklabels=car_corr.columns,     #Definimimos el nombre para el eje x
              yticklabels=car_corr.columns,     #Definimimos el nombre para el eje y
              cmap='coolwarm'                   #Definimos los colores
              ) 
plt.show()                                      #Lanzamos la grafica
print('\n',format(' Matriz de correlacion datos numericos ','*^82'),'\n')
correr = pg.pairwise_corr(df, method='spearman')                    #defiimos el metodo por el cual se hara la matriz de correlacion 
print(correr.sort_values(by=['p-unc'])[['X','Y','n','r','p-unc']])  #En consola mostramos la matriz


plt.scatter(df['price'],df['year'],c="green", alpha=0.2)   #Cargamos las columnas correlacionadas
plt.xticks(rotation=60)                                    #Rotamos las etiquetas
plt.xlabel("price")                                        #Etiqueta del eje x
plt.ylabel("year")                                         #Etiqueta del eje x
plt.title("Year-Price")                                    #Etiqueta del titulo
plt.show()                                                 #Lanzamos la grafica

plt.scatter(df['mileage'],df['vol_engine'],c="green", alpha=0.2)   #Cargamos las columnas correlacionadas
plt.xticks(rotation=60)                                            #Rotamos las etiquetas
plt.xlabel("mileage")                                              #Etiqueta del eje x
plt.ylabel("vol_engine")                                           #Etiqueta del eje x
plt.title("mileage-vol_engine")                                    #Etiqueta del titulo
plt.show()                                                         #Lanzamos la grafica


columnas_scatter = ['price', 'fuel', 'year']               #Defininimos un arreglo con el nombre de columnas que estarn correlacionadas
for i in columnas_scatter:                                 #Inicamos for con el nombre de estas columnas
    plt.scatter(df['mark'],df[i],c="green", alpha=0.2)     #Cargamos las etiquetas
    plt.xticks(rotation=60)                                #Rotamos las etiquegtas 
    plt.xlabel("mark")                                     #Etiqueta del eje x
    plt.ylabel(i)                                          #Etiqueta del eje y
    plt.title("Mark-"+i)                                   #Etiqueta del titulo
    plt.show()                                             #Lanzamos la grafica
print(df.info())
#---------Interprestacion de datos END------#

#---------Creacion del clasificador Begin------#
X = df.iloc[:, [3,10,11,14,15]]                                                             #Seleccion de variables descriptivas
Y = df.iloc[:,12]                                                                           #Seleccion de variable objetivo
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.70, random_state= 0)  #Separacion de datos entrenamiento
arbol = DecisionTreeClassifier(max_depth=5,                                                 #Maximo de niveles
                                class_weight="balanced",                                    #Balanceamos datos
                                )
arbol_precios = arbol.fit(x_train,y_train)                                                  #Entrenamos el modelo
fig = plt.figure(figsize=(140,20))                                                          #Definimos tamaño de la figura
tree.plot_tree(arbol_precios, feature_names=list(X.columns.values),                         #Cargamos la figura
                            class_names=list(Y.values), filled=True, fontsize=10)           #Defininimos letra
#plt.show()

fig.savefig("figuraExamen.png")                                                             #Guardamos en la computadora la img
y_pred =arbol_precios.predict(x_test)                                                       #Hacemos la prediccion con los datos de prueba
matriz = confusion_matrix(y_test, y_pred)                                                   #Creamos la matriz de confucion
print(matriz)                                                                               #Lanzamos la matriz

precicion =  np.sum(matriz.diagonal())/np.sum(matriz)                                       #Calculamos precision/exactitud
print("Precision: ", precicion)                                                             #Lanzamos resultadi 

                                                  
exactitud = accuracy_score(y_test, y_pred)                                                  #Metodo sklearn para precision
print("Precision: ", exactitud)                                                             #Lanzamos resultado

#---------Creacion del clasificador END------#


#---------Hipotesis Begin------#
age_groups = pd.cut(df['mileage'],                                              #Seccionamos por mileage                                                          
                        bins=[np.NINF, 22625, 45250,67875,90500,113125,135750,158375,181000,203625,226250,248875,271500, np.inf])
pd.crosstab(age_groups, df['mark'], values=df['price'], aggfunc='mean').plot()  #Grafica de la media de precios segun el combustible a lo largo de los años
plt.show()                                                                      #lanzamos la grafica

viaje = pd.cut(df['mileage'], bins=[203625, np.inf])                            #Seleccionamos por viaje
pd.crosstab(viaje, df['fuel']).plot.bar()                                       #Contamos segun el combustible
plt.show()                                                                      #Lanzamos la grafica
#---------Hipotesis End------#

#---------Testeo Begin------#
datos = [                                       
    {'year': 2010, 'mark_transform': 1, 'fuel_transform': 0, 'mileage_norm': 256000, 'vol_engine_norm': 1968},
    {'year': 2005, 'mark_transform': 5, 'fuel_transform': 1, 'mileage_norm': 167933, 'vol_engine_norm': 998},
    {'year': 1990, 'mark_transform': 6, 'fuel_transform': 3, 'mileage_norm': 148933, 'vol_engine_norm': 1368},
    {'year': 2018, 'mark_transform': 7, 'fuel_transform': 2, 'mileage_norm': 152000, 'vol_engine_norm': 1598},
    {'year': 2000, 'mark_transform': 8, 'fuel_transform': 4, 'mileage_norm': 98258,  'vol_engine_norm': 1598},
    {'year': 2005, 'mark_transform': 9, 'fuel_transform': 5, 'mileage_norm': 109330, 'vol_engine_norm': 1398}
]
df_testeo = pd.DataFrame(datos)                                                             #Creamos un data set con los datos de prueba
scale= StandardScaler()                                                                     #cargamos el metodo para estandarizar
xnorm2= df_testeo[['mileage_norm','vol_engine_norm']]                                       #aasignamos las columnas a normalizar
scaledx2 = scale.fit_transform(xnorm2)                                                      #Aplicamos la normalizacion    
auxnorm2 = pd.DataFrame(scaledx2, columns=['mileage_norm','vol_engine_norm'])               #Guardamos en data set auxiliar
df_testeo[['mileage_norm','vol_engine_norm']] = auxnorm2[['mileage_norm','vol_engine_norm']]#Guardamos en data set original
y_predtesteo = arbol_precios.predict(df_testeo)                                             #Ejecutamos la prediccion
df_testeo['resultadols'] = pd.DataFrame(y_predtesteo)                                       #Almacenamos el resulatdo
y_proba = arbol_precios.predict_proba(df_testeo.drop(['resultadols'], axis=1))              #Cargamos datos para la probabilidad
porcentaje = np.max(arbol_precios.predict_proba(df_testeo.drop(['resultadols'], axis=1)), axis=1)   #Ejecutamos la probabilidad
df_testeo['porcentaje'] = pd.DataFrame(porcentaje)                                                  #Guardamos el resultado
print(df_testeo)                                                                                    #Mostramos datos finales
#df.to_csv('car_power_bi.csv')
#---------Testeo End------#