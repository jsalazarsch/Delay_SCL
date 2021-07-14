# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:10:19 2021

@author: Joaquin Salazar Schlotterbeck - realizado en Python 3.8
"""
#pip install imbalanced-learn

#Manipulacion de datos
import numpy as np
import pandas as pd
import datetime as datetime
import calendar #Revisar dsps

#Modelamiento
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
    
#Muestreo - Submuestreo
from imblearn.under_sampling import RandomUnderSampler, NearMiss

#Evaluación del modelo
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score,recall_score,f1_score,roc_auc_score


#Importar bases usadas
df=pd.read_csv('C:\Users\jsala\OneDrive - Universidad Adolfo Ibanez\PEGA\LATAM\Airport_Delay\dataset_SCL.csv')

File = pd.ExcelFile('C:\Users\jsala\OneDrive - Universidad Adolfo Ibanez\PEGA\LATAM\Airport_Delay\CONT.xlsx')
df_cont=File.parse('CONT')
#Como distribuyen los datos

print('Información General: \n Dimensiones del DF: \n',df.shape,'\n Nombre de las variables: \n',df.keys(),'\n Información del DF: \n',df.info(),'\n')


print('Tablas de Frecuencia de las variables: \n')
print(df.keys())
def month(x):
  return calendar.month_name[x]
df['MES'] = df['MES'].apply(month)
df['TIPOVUELO']=np.where(df['TIPOVUELO']=='I','Internacional','Nacional')


all_var=df.keys()
for a in all_var:
    #print('Dimensiones de',i,':\n',df[i].shape,'\n Descripción de Variable ',i,':\n',df[i].describe(),'\n Conteo de Valores para la variable',i,':\n\n',df[i].value_counts(),'\n')
    frec=pd.value_counts(df[a])
    frec_df=pd.DataFrame(frec)
    frec_df.columns = ['Frec_abs']
    frec_df['Frec_rel_%']=100*frec_df['Frec_abs']/len(df)
    frec_rel_val=frec_df['Frec_rel_%'].values
    acum=[]
    valor_acum=0
    for i in frec_rel_val:
        valor_acum=valor_acum+i
        acum.append(valor_acum)
    frec_df['Frec_rel_%_acum']=acum
    print('Tabla de frecuencia de',a,':\n',frec_df,'\n')
    del(a,acum,frec,frec_df,frec_rel_val,i,valor_acum)
print('Al ser categoricas se usaron tablas de frecuencia para poder ver como se distribuian, se pudo obserbar \n una grn concentracion en vuelos nacionales de LATAM, a partir de este analisis pudimos descartar la variable de origen ya que todos provenian de SCL.\n')


#CREACION DE VARIABLES
#1.1 - TEMPORADA ALTA
#-Se crean las variables de las fechas para comparar si una fecha es temporada alta o no
tempa=(pd.to_datetime("2017-12-15 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))
tempb=(pd.to_datetime("2017-03-03 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))
tempc=(pd.to_datetime("2017-07-15 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))
tempd=(pd.to_datetime("2017-07-31 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))
tempe=(pd.to_datetime("2017-09-11 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))
tempf=(pd.to_datetime("2017-09-30 00:00:00").strftime('%Y-%m-%d %H:%M:%S'))

#Se hace la comparación
df['temporada_alta']=np.where((((df['Fecha-I']>= tempa))|((df['Fecha-I'] <= tempb))|(((df['Fecha-I']>=tempc))&((df['Fecha-I']<=tempd)))|(((df['Fecha-I']>=tempd))&((df['Fecha-I']<=tempe)))),1,0)


#1.2 - DIFERENCIA DE MINUTOS ENTRE PROGRAMADO Y OPERACION
#Se pasan las variables Fecha-I y Fecha-O a date time para trabajar con las horas
df['Fecha-I']=pd.to_datetime(df['Fecha-I'], format="%Y-%m-%d %H:%M:%S")
df['Fecha-O']=pd.to_datetime(df['Fecha-O'], format="%Y-%m-%d %H:%M:%S")
#Se crea la variable dif_min, como observo valores negativos (llegaron antes) determino el delay maximo como punto de referencia y homologar el formato del tiempo a minutos
df['dif_min']=(df['Fecha-O']-df['Fecha-I'])
#Obtenemos el mayor delay para usarlo de referencia para los que llegaron antes y lo pasamos a minutos
maxmin=(df['dif_min'].max().seconds/600)
#ya creado el mayor delay como punto de referencia, se asignan los valores en minutos a la variable (quedan en floats positivos y negativos para distinguir la data y no quedaran outliers con los que llegaron antes)
df['dif_min']=np.where((df['dif_min'].dt.seconds.astype('float')/60)<=maxmin,(df['dif_min'].dt.seconds.astype('float')/60),(df['dif_min'].dt.seconds.astype('float')/60-1440))#ARREGLAR MINUTOS
#1.3 - ATRASO > A 15 MINUTOS OK
df['atraso_15']=np.where((df['dif_min'])>15,1,0)
#1.4 - PERIODO DEL DÍA 

def periodo(hora_dia):
    
    if hora_dia>=5 and hora_dia<12:
       return 'Manana'
    elif hora_dia>=12 and hora_dia<19:
        return'Tarde'
    else:
        return 'Noche'
periodod = []

for item in df['Fecha-I']:
    periodod.append(periodo(int(item.strftime('%H'))))
periodod=pd.DataFrame(periodod)
df=pd.concat([df,periodod],axis=1)

df = df.rename(columns={0:'periodo_dia'})

#EXPORTAR ARCHIVO SYNTHETIC_FEATURES
ruta = "C:/Users/jsala/OneDrive - Universidad Adolfo Ibanez/PEGA/LATAM/synthetic_features.csv"
df.to_csv(ruta)

#Eliminar variables que no se usarán /están duplicadas y no sirven
del(tempa,tempb,tempc,tempd,tempe,tempf,periodod,maxmin,item)
del(df['Ori-I'],df['Ori-O'])

#Creación de Variables - Cambios en los Vuelos (Avion, Destino, Origen, Empresa)
df['CambioVlo']=np.where((df['Vlo-I']!=df['Vlo-O']),1,0)
df['CambioDes']=np.where((df['Des-I']!=df['Des-O']),1,0)
df['CambioEmp']=np.where((df['Emp-I']!=df['Emp-O']),1,0)


#Categorizar Variables
df['MES']=pd.Categorical(df['MES'],categories=['January','February','March','April','May','June','July','August','September','October','November','December'],ordered=True)
df['DIA']=pd.Categorical(df['DIA'],categories=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],ordered=True)
df['DIANOM']=pd.Categorical(df['DIANOM'],categories=['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo'],ordered=True)
df['AÑO']=pd.Categorical(df['AÑO'],categories=[2017,2018],ordered=True)
df['Vlo-I']=df['Vlo-I'].astype('category')
df['Des-I']=df['Des-I'].astype('category')
df['Emp-I']=df['Emp-I'].astype('category')
df['Vlo-O']=df['Vlo-O'].astype('category')
df['Des-O']=df['Des-O'].astype('category')
df['Emp-O']=df['Emp-O'].astype('category')
df['TIPOVUELO']=df['TIPOVUELO'].astype('category')
df['SIGLAORI']=df['SIGLAORI'].astype('category')
df['SIGLADES']=df['SIGLADES'].astype('category')


#COMO SE COMPONE EL ATRASO, QUE VARIABLES ESPERARIAS MAS QUE INFLUYEN 
atrasos=sum(df['atraso_15']==1)
total_vuelos=len(df)
tasa_atrasos=atrasos/total_vuelos
print('Tasa global de atrasos:\n',tasa_atrasos,'\n')



def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

graf=['SIGLADES','OPERA','MES','DIANOM','temporada_alta','TIPOVUELO','CambioVlo','CambioEmp','CambioDes']
high_val=pd.DataFrame()

for x in graf:
    
    compos = df['atraso_15'].groupby(df[x]).apply(get_stats).unstack()
    compos = compos.sort_values('mean',ascending=False)
    compos['Variable']=str(x)
    high_val= high_val.append(compos)
    print(compos)

high_val=high_val[high_val['mean']>=tasa_atrasos]
print('\n Valores sobre la tasa global de atrasos:\n', high_val,'\n')
print('3. Después de ver como se comporta la tassa de atraso, creo que las variables que más \n debiesen influir son las que están por sobre la tasa global de atrasos \n en este caso son:\n',high_val['Variable'].unique(),'\n')

df=df.merge(df_cont, on='Des-I', how='left')
del(df['SIGLAORI'],df['SIGLADES_y'],df_cont,atrasos,total_vuelos,x,graf,compos,all_var,File,tasa_atrasos)

#Vectorizar Variables Categoricas
df2=df[['atraso_15','SIGLADES_x','OPERA','DIANOM','temporada_alta','TIPOVUELO','CambioVlo','CambioEmp','CambioDes']]
all_var=df2.keys()
df_clean=pd.DataFrame()
DU1=pd.get_dummies(df2['SIGLADES_x'],prefix='SIGLADES_')
DU2=pd.get_dummies(df2['OPERA'],prefix='OPERA_')
DU3=pd.get_dummies(df2['DIANOM'],prefix='DIANOM_')
DU4=df2['temporada_alta']
DU5=pd.get_dummies(df2['TIPOVUELO'],prefix='TIPOVUELO_')
DU6=df2['CambioVlo']
DU7=df2['CambioEmp']
DU8=df2['CambioDes']
#DU9=pd.get_dummies(df2['PAIS'],prefix='PAIS_')
#DU10=pd.get_dummies(df2['CONT'],prefix='CONT_')
df_clean=pd.concat([DU1,DU2,DU4,DU5,DU7],axis=1)

print('Submuestreo:')
#NearMiss. Elimina las muestras más cercanas de la clase más representada
nm = NearMiss()
dataNm, targetNm = nm.fit_resample(df_clean,df2['atraso_15'])
PuntualNm = targetNm.sum()
AtrasadoNm = targetNm.shape[0]- PuntualNm
print('Vuelos Puntuales: ', PuntualNm, ' , Vuelos Atrasados: ', AtrasadoNm )

#Asignar bases balanceadas a las variables para los modelos
X=dataNm
y=targetNm

#MODELO 1 - SVR
#Se separan los datasets de entrenamiento y de prueba
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Se asigna el modelo
svr=SVR()
#Se entrena el modelo
svr.fit(X_train,y_train)
#Predecimos con el dataset de prueba
y_predsvr=svr.predict(X_test)
#Resultados del modelo
print('Evaluacion del modelo 1 - SVR: \n Precision: ',svr.score(X_train,y_train),'\n')
#MODELO 2 - REGRESION POLINOMIAL
#Se separan los datasets de entrenamiento y de prueba
X2_trainp,X2_testp,y2_trainp,y2_testp= train_test_split(X,y,test_size=0.2)
#Definir grado del polinomio
poli_reg=PolynomialFeatures(degree=2)
#Se trasnforma las caracteristicas existentes en caracteristicas de mayor grado
X2_train_poli= poli_reg.fit_transform(X2_trainp)
X2_test_poli=poli_reg.fit_transform(X2_testp)
#Definimos el modelo
pr=linear_model.LinearRegression()
#Entrenamos el modelo
pr.fit(X2_train_poli,y2_trainp)
#Realizo prediccion
Y2_predpoli=pr.predict(X2_test_poli)
#Resultados del modelo
print('Evaluacion del modelo 2 - REGRESION POLINOMIAL: \n Precision: ',pr.score(X2_train_poli,y2_trainp),'\n')
#Intercepto:
print('Intercepto: ',pr.intercept_)
#Pendiente
print('Coeficientes: ',pr.coef_,'\n')
#MODELO 3 - RANDOM FOREST
#Se separan los datasets de entrenamiento y de prueba
X3_train,X3_test,y3_train,y3_test=train_test_split(X,y,test_size=0.2)
#Definimos el modelo
bar=RandomForestRegressor(n_estimators=300, max_depth=8)
#Entrenamos el modelo
bar.fit(X3_train,y3_train)
#Realizo prediccion
y3_predbar=bar.predict(X3_test)
#Resultados del modelo
print('Evaluacion del modelo 3 - RANDOM FOREST: \n Precision: ',bar.score(X3_train,y3_train),'\n')

#Separar la base de entrenamiento y prueba - Modelo 3
X4_train,X4_test,y4_train,y4_test=train_test_split(X,y,test_size=0.2)

#Seleccionar modelo (Regresion Logistica) - Entrenar y predecir
algoritmo=LogisticRegression()
algoritmo.fit(X4_train,y4_train)
y4_pred=algoritmo.predict(X4_test)



#Metricas - Reg. Logistica
matriz=confusion_matrix(y4_test,y4_pred)
exactitud=accuracy_score(y4_test,y4_pred)
precision=precision_score(y4_test,y4_pred)
sensibilidad=recall_score(y4_test,y4_pred)
puntaje=f1_score(y4_test,y4_pred)
roc_auc=roc_auc_score(y4_test,y4_pred)


print('INFORMACION DEL MODELO 4(EXTRA): REGRESION LOGISTICA')
print('Matriz de confusión: \n',matriz)
print('Precision del modelo: ',precision)
print('Exactitud del modelo: ',exactitud)
print('Sensibilidad del modelo: ',sensibilidad)
print('Puntaje F1 del modelo: ',puntaje)
print('Curva ROC - AUC del modelo: ',roc_auc)

print('\n Al ser modelos de regresión usé R**2 para evaluar los modelos, lo usé porque me sirve para ver \n que tan alejados se encuentran los valores reales con respecto a los valores predichos\n los valores que mas influyeron en el modelo eran los cambios de empresa y si el vuelo era internacional\n Podría mejorar al performance teniendo datos como el porte del avión, las condiciones climaticas de las fechas de vuelo \n y saber si los pasajeros vienen de alguna escala según el levantamiento que realicé con alguien del negocio.')