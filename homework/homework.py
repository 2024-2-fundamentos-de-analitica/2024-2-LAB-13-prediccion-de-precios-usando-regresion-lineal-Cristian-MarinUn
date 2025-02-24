import os
import gzip
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def cargar_datos_comprimidos(ruta):
    return pd.read_csv(ruta, compression='zip')

def procesar_datos(df):
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], inplace=True, errors='ignore')
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
    return df

def construir_modelo(datos):
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = list(set(datos.columns) - set(columnas_categoricas))
    transformador = ColumnTransformer([
        ("num", StandardScaler(), columnas_numericas),
        ("cat", OneHotEncoder(), columnas_categoricas)
    ], remainder='passthrough')
    return Pipeline([
        ('preprocesado', transformador),
        ('reduccion_dimensionalidad', PCA()),
        ('seleccion_caracteristicas', SelectKBest(f_classif)),
        ('modelo_svc', SVC(kernel="rbf", max_iter=-1, random_state=42))
    ])

def configurar_busqueda(pipeline):
    parametros = {
        "reduccion_dimensionalidad__n_components": [0.8, 0.9, 0.95, 0.99],
        "seleccion_caracteristicas__k": [10, 20, 30],
        "modelo_svc__C": [0.1, 1, 10],
        "modelo_svc__gamma": [0.1, 1, 10]
    }
    return GridSearchCV(pipeline, param_grid=parametros, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2, refit=True)

def almacenar_modelo(ruta, modelo):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

def calcular_metricas(nombre, y_real, y_predicho):
    return {
        'type': 'metrics',
        'dataset': nombre,
        'precision': precision_score(y_real, y_predicho, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_real, y_predicho),
        'recall': recall_score(y_real, y_predicho, zero_division=0),
        'f1_score': f1_score(y_real, y_predicho, zero_division=0)
    }

def calcular_matriz_confusion(nombre, y_real, y_predicho):
    matriz = confusion_matrix(y_real, y_predicho)
    return {
        'type': 'cm_matrix',
        'dataset': nombre,
        'true_0': {"predicted_0": int(matriz[0, 0]), "predicted_1": int(matriz[0, 1])},
        'true_1': {"predicted_0": int(matriz[1, 0]), "predicted_1": int(matriz[1, 1])}
    }

def ejecutar_proceso():
    carpeta_datos = "./files/input/"
    modelo_ruta = "./files/models/model.pkl.gz"
    
    df_test = cargar_datos_comprimidos(os.path.join(carpeta_datos, 'test_data.csv.zip'))
    df_train = cargar_datos_comprimidos(os.path.join(carpeta_datos, 'train_data.csv.zip'))
    
    df_test = procesar_datos(df_test)
    df_train = procesar_datos(df_train)
    
    x_test, y_test = df_test.drop(columns=['default']), df_test['default']
    x_train, y_train = df_train.drop(columns=['default']), df_train['default']
    
    modelo_pipeline = construir_modelo(x_train)
    modelo_final = configurar_busqueda(modelo_pipeline)
    modelo_final.fit(x_train, y_train)
    
    almacenar_modelo(modelo_ruta, modelo_final)
    
    y_train_predicho = modelo_final.predict(x_train)
    y_test_predicho = modelo_final.predict(x_test)
    
    metricas_resultado = [
        calcular_metricas('train', y_train, y_train_predicho),
        calcular_metricas('test', y_test, y_test_predicho),
        calcular_matriz_confusion('train', y_train, y_train_predicho),
        calcular_matriz_confusion('test', y_test, y_test_predicho)
    ]
    
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as archivo:
        for resultado in metricas_resultado:
            archivo.write(json.dumps(resultado) + "\n")

if __name__ == "__main__":
    ejecutar_proceso()
