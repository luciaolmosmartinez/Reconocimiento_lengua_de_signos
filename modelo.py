import pandas as pd
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELOS_DIR = BASE_DIR / "modelos"
RESULTADOS_DIR = BASE_DIR / "resultados"

MODELOS_DIR.mkdir(exist_ok=True)
RESULTADOS_DIR.mkdir(exist_ok=True)

RUTA_DECISION_TREE = MODELOS_DIR / "decision_tree.pckl"
RUTA_RANDOM_FOREST = MODELOS_DIR / "random_forest.pckl"
RUTA_RED_NEURONAL = MODELOS_DIR / "red_neuronal.pckl"
RUTA_KNN = MODELOS_DIR / "knn.pckl"
RUTA_SVM = MODELOS_DIR / "svm.pckl"

def generar_matriz_confusion(nombre_fichero, y_test, y_pred):
    clases = sorted(np.unique(y_test))

    matriz = confusion_matrix(y_test, y_pred, labels=clases)

    fig, ax = plt.subplots(figsize=(12, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=clases
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True
    )

    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(str(nombre_fichero), dpi=300)
    plt.show()

def calcular_cmc(modelo, nombre_fichero, X_test, y_test, max_k=None):
    y_test = np.asarray(y_test)

    if hasattr(modelo, "predict_proba"):
        puntuaciones = modelo.predict_proba(X_test)
    elif hasattr(modelo, "decision_function"):
        puntuaciones = modelo.decision_function(X_test)
    else:
        raise ValueError("El modelo no permite obtener probabilidades ni puntuaciones por clase.")

    clases = modelo.classes_

    if max_k is None:
        max_k = len(clases)

    max_k = min(max_k, len(clases))

    # Ordena las clases de mayor a menor probabilidad/puntuación
    indices_ordenados = np.argsort(puntuaciones, axis=1)[:, ::-1]
    clases_ordenadas = clases[indices_ordenados]

    valores_cmc = []

    for k in range(1, max_k + 1):
        aciertos_top_k = np.any(clases_ordenadas[:, :k] == y_test[:, None], axis=1)
        valores_cmc.append(np.mean(aciertos_top_k))

    k = np.arange(1, max_k + 1)
    valores_cmc = np.array(valores_cmc)

    plt.figure(figsize=(8, 6))
    plt.plot(k, valores_cmc, marker="o")
    plt.xlabel("k")
    plt.ylabel("CMC(k)")
    plt.title("Curva CMC")
    plt.xticks(k)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(nombre_fichero), dpi=300)
    plt.show()
    

def separacion_datos(datos):
    with open(datos, 'r', encoding='utf-8') as f:
        lineas_limpias = [linea.strip().strip('"') for linea in f.readlines()]
    
    columnas = lineas_limpias[0].split(',')
    datos_filas = [linea.split(',') for linea in lineas_limpias[1:]]

    signos_df = pd.DataFrame(datos_filas, columns=columnas)

    for col in signos_df.columns:
        if col != 'Signo':
            signos_df[col] = pd.to_numeric(signos_df[col])

    y = signos_df['Signo']
    x = signos_df.drop(columns=['Signo']) #demasiadas columnas como para ponerlas

    return train_test_split(x,y, test_size=0.2, random_state=42, stratify=y) #random state para controlar aleatoriedad, ahora mismo se reparten los datos siempre igual entre los dos grupos de dataset

def entrenar(entrenar_x,entrenar_y):
    # decision_tree
    decision_tree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5)
    decision_tree.fit(entrenar_x,entrenar_y)
    with open(RUTA_DECISION_TREE, 'wb') as f:
        pickle.dump(decision_tree, f)

    # random_forest
    random_forest = RandomForestClassifier(n_estimators=200, criterion='entropy', max_features='log2', min_samples_split=10, max_depth=100)
    random_forest.fit(entrenar_x,entrenar_y)
    with open(RUTA_RANDOM_FOREST, 'wb') as f:
        pickle.dump(random_forest, f)

    # red_neuronal
    red_neuronal = MLPClassifier(hidden_layer_sizes=(150,), solver='lbfgs', max_iter=1000, learning_rate='adaptive')
    red_neuronal.fit(entrenar_x,entrenar_y)
    with open(RUTA_RED_NEURONAL, 'wb') as f:
        pickle.dump(red_neuronal, f)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7, p=4)
    knn.fit(entrenar_x,entrenar_y)
    with open(RUTA_KNN, 'wb') as f:
        pickle.dump(knn, f)

    # SVM
    svm = SVC(C=6, kernel='poly', degree=6, probability=True)
    svm.fit(entrenar_x,entrenar_y)
    with open(RUTA_SVM, 'wb') as f:
        pickle.dump(svm, f)

def test(validar_x,validar_y):
    # decision_tree
    try:
        ruta_decision_tree = RUTA_DECISION_TREE
        with open(ruta_decision_tree, 'rb') as f:
            decision_tree = pickle.load(f)
        print(f"-> Modelo cargado correctamente desde: {ruta_decision_tree}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta_decision_tree}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")
        return
    predicciones_decision_tree = decision_tree.predict(validar_x)
    generar_matriz_confusion(RESULTADOS_DIR / "matriz_confusion_decision_tree.png", validar_y, predicciones_decision_tree)
    calcular_cmc(decision_tree, RESULTADOS_DIR / "cmc_decision_tree.png", validar_x, validar_y, 5)

    # random_forest
    try:
        ruta_random_forest = RUTA_RANDOM_FOREST
        with open(ruta_random_forest, 'rb') as f:
            random_forest = pickle.load(f)
        print(f"-> Modelo cargado correctamente desde: {ruta_random_forest}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta_random_forest}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")
        return
    predicciones_random_forest = random_forest.predict(validar_x)
    generar_matriz_confusion(RESULTADOS_DIR / "matriz_confusion_random_forest.png", validar_y, predicciones_random_forest)
    calcular_cmc(random_forest, RESULTADOS_DIR / "cmc_random_forest.png", validar_x, validar_y, 5)

    # red_neuronal
    try:
        ruta_red_neuronal = RUTA_RED_NEURONAL
        with open(ruta_red_neuronal, 'rb') as f:
            red_neuronal = pickle.load(f)
        print(f"-> Modelo cargado correctamente desde: {ruta_red_neuronal}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta_red_neuronal}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")
        return
    predicciones_red_neuronal = red_neuronal.predict(validar_x)
    generar_matriz_confusion(RESULTADOS_DIR / "matriz_confusion_red_neuronal.png", validar_y, predicciones_red_neuronal)
    calcular_cmc(red_neuronal, RESULTADOS_DIR / "cmc_red_neuronal.png", validar_x, validar_y, 5)
    

    # KNN
    try:
        ruta_knn = RUTA_KNN
        with open(ruta_knn, 'rb') as f:
            knn = pickle.load(f)
        print(f"-> Modelo cargado correctamente desde: {ruta_knn}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta_knn}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")
        return
    predicciones_knn = knn.predict(validar_x)
    generar_matriz_confusion(RESULTADOS_DIR / "matriz_confusion_knn.png", validar_y, predicciones_knn)
    calcular_cmc(knn, RESULTADOS_DIR / "cmc_knn.png", validar_x, validar_y, 5)


    # SVM
    try:
        ruta_svm = RUTA_SVM
        with open(ruta_svm, 'rb') as f:
            svm = pickle.load(f)
        print(f"-> Modelo cargado correctamente desde: {ruta_svm}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta_svm}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")
        return
    predicciones_svm = svm.predict(validar_x)
    generar_matriz_confusion(RESULTADOS_DIR / "matriz_confusion_svm.png", validar_y, predicciones_svm)
    calcular_cmc(svm, RESULTADOS_DIR / "cmc_svm.png", validar_x, validar_y, 5)
    

    fallos_decision_tree = (predicciones_decision_tree != validar_y.values).sum()
    acierto_decision_tree = accuracy_score(validar_y, predicciones_decision_tree) * 100

    fallos_random_forest = (predicciones_random_forest != validar_y.values).sum()
    acierto_random_forest = accuracy_score(validar_y, predicciones_random_forest) * 100

    fallos_red_neuronal = (predicciones_red_neuronal != validar_y.values).sum()
    acierto_red_neuronal = accuracy_score(validar_y, predicciones_red_neuronal) * 100

    fallos_knn = (predicciones_knn != validar_y.values).sum()
    acierto_knn = accuracy_score(validar_y, predicciones_knn) * 100
    
    fallos_svm = (predicciones_svm != validar_y.values).sum()
    acierto_svm = accuracy_score(validar_y, predicciones_svm) * 100
    
    print("--- RESULTADOS DEL MODELO ---")
    print(f"Total de imágenes evaluadas: {len(validar_y)}")
    print(f"Cantidad de predicciones FALLADAS decision tree: {fallos_decision_tree}")
    print(f"Cantidad de predicciones ACERTADAS: {len(validar_y) - fallos_decision_tree}")
    print(f"Porcentaje de éxito (Accuracy): {acierto_decision_tree:.2f}%")
    print("-------------------------------------------------")
    print(f"Cantidad de predicciones FALLADAS random forest: {fallos_random_forest}")
    print(f"Cantidad de predicciones ACERTADAS: {len(validar_y) - fallos_random_forest}")
    print(f"Porcentaje de éxito (Accuracy): {acierto_random_forest:.2f}%")
    print("-------------------------------------------------")
    print(f"Cantidad de predicciones FALLADAS red neuronal: {fallos_red_neuronal}")
    print(f"Cantidad de predicciones ACERTADAS: {len(validar_y) - fallos_red_neuronal}")
    print(f"Porcentaje de éxito (Accuracy): {acierto_red_neuronal:.2f}%")
    print("-------------------------------------------------")
    print(f"Cantidad de predicciones FALLADAS knn: {fallos_knn}")
    print(f"Cantidad de predicciones ACERTADAS: {len(validar_y) - fallos_knn}")
    print(f"Porcentaje de éxito (Accuracy): {acierto_knn:.2f}%")
    print("-------------------------------------------------")
    print(f"Cantidad de predicciones FALLADAS svm: {fallos_svm}")
    print(f"Cantidad de predicciones ACERTADAS: {len(validar_y) - fallos_svm}")
    print(f"Porcentaje de éxito (Accuracy): {acierto_svm:.2f}%")

def usar_modelo(datos_originales, modelo):
    #datos = datos_originales.reshape(1, -1)

    try:
        ruta = ""
        if modelo == "arbol":
            ruta = RUTA_DECISION_TREE
        elif modelo == "bosque":
            ruta = RUTA_RANDOM_FOREST
        elif modelo == "red neuronal" or modelo == "red":
            ruta = RUTA_RED_NEURONAL
        elif modelo == "knn":
            ruta = RUTA_KNN
        elif modelo == "svm":
            ruta = RUTA_SVM
        else:
            print(f"ERROR: Modelo no reconocido: {modelo}")
            print("Modelos disponibles: arbol, bosque, red neuronal, knn, svm")
            return None

        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        datos = pd.DataFrame(
            [datos_originales],
            columns=modelo.feature_names_in_
        )
        probabilidades = modelo.predict_proba(datos)
        #print(probabilidades)
        return modelo.predict(datos)[0]
    except FileNotFoundError:
        print(f"ERROR: No se encontró ningún archivo .pckl en {ruta}.")
        print("Por favor, ejecuta primero el script de entrenamiento correspondiente.")    

def main():
    # Código para determinar si hace una cosa o la otra
    print("============================")
    print("           MODELO           ")
    print("============================")
    print("Selecciona una opción:")
    print("[e] - Entrenar todos los modelos")
    print("[t] - Realizar el test de los modelos")
    print("[q] - Salir")
    print("----------------------------")

    opcion = input("Introduce tu elección: ").lower()
    if opcion == 'q':
        print("Saliendo...")
        exit()
    else:
        datos = BASE_DIR / input("Introduce el nombre de tu csv: ")
        entrenar_x, validar_x, entrenar_y, validar_y = separacion_datos(datos)
        lineas_csv_entrenamiento = entrenar_x.index + 2
        lineas_csv_validacion = validar_x.index + 2

        print(lineas_csv_entrenamiento.tolist())
        print('\n\nTEST\n')
        print(lineas_csv_validacion.tolist())
        if opcion == 'e':
            entrenar(entrenar_x,entrenar_y)
        elif opcion == 't':
            test(validar_x,validar_y)
        else:
            print("Opción no válida. Inténtalo de nuevo.")
 

# PRINCIPIO DEL CODIGO
if __name__ == "__main__":
    main()