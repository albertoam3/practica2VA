import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


num_senal1 = ['0', '1', '2', '3', '4', '5', '7', '8', '9', '10', '15', '16']
num_senal2 = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
num_senal3 = ['14']
num_senal4 = ['17']
num_senal5 = ['13']
num_senal6 = ['38']

class_features = [[], [], [], [], [], [], []]
class_labels = [[], [], [], [], [], [], []]

gnbs = []
ejercicio = True

def ejercicio_check(ejer):
    global ejercicio
    ejercicio = ejer

def gnb_func(X_val, Y_val):
    # Inicializar y ajustar el clasificador Bayesiano con Gaussianas
    gnb = GaussianNB()
    gnb.fit(X_val, Y_val)
    gnbs.append(gnb)

def apply_LDA(X_val, y_val):
    hog_matrix = np.array(X_val)
    lda = LDA()
    lda.fit(hog_matrix, y_val)
    # Reducir la dimensionalidad de los datos de entrenamiento con LDA
    return lda.transform(hog_matrix)


def unique_classifier(X_val, classifier):
    etiquetas = classifier.predict(X_val)
    return etiquetas


def multiclass_classifier(X_val):
    # Obtener las probabilidades de pertenecer a cada clase para cada clasificador binario
    probabilities = []
    etiquetas = []
    for gnb in gnbs:
        probabilities.append(gnb.predict_proba(X_val))
        etiquetas.append(gnb.predict(X_val))

    resultado = []

    for j, prob in enumerate(probabilities):
        ind = np.argmax(prob, axis=1)
        indices_y_probabilidades = list(zip(ind, np.max(prob, axis=1)))
        for i, (indice, probabilidad) in enumerate(indices_y_probabilidades):

            if len(resultado) > i and (
                    (probabilidad > resultado[i][1] and indice != 0) or (resultado[i][0] == 0 and indice != 0)):
                if indice == 0:
                    resultado[i] = [indice, probabilidad]
                else:
                    resultado[i] = [j + 1, probabilidad]
            elif len(resultado) <= i:
                if indice == 0:
                    resultado.append([indice, probabilidad])
                else:
                    resultado.append([j + 1, probabilidad])

    resultado_final = []
    porcentaje_final = []
    for i, p in resultado:
        resultado_final.append(i)
        porcentaje_final.append(p)
    return resultado_final, porcentaje_final


def hog(image):
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_vector = hog.compute(image)
    return hog_vector


def expand_detected_regions(regions, gray_image, original_image, datos, expand_factor=1.2):
    expanded_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            new_x = max(0, int(x - (expand_factor - 1) / 2 * w))
            new_y = max(0, int(y - (expand_factor - 1) / 2 * h))
            new_w = min(gray_image.shape[1], int(w * expand_factor))
            new_h = min(gray_image.shape[0], int(h * expand_factor))
            encontrado = False
            imagen_recortada = original_image[new_y:new_y + new_h, new_x:new_x + new_w]
            imagen_recordata_escala = resize_regions(imagen_recortada)

            for dato in datos:

                if comparar_rectangulos(new_x, new_y, new_x + new_w, new_y + new_h, int(dato[1]), int(dato[2]),
                                        int(dato[3]), int(dato[4])):
                    encontrado = True
                    repetido = False
                    for reg in expanded_regions:
                        if comparar_rectangulos(reg[0], reg[1], reg[2] + reg[0], reg[3] + reg[1], new_x,
                                                new_y, new_w + new_x, new_h + new_y):
                            repetido = True
                    if not repetido:
                        n = -1
                        if dato[5] in num_senal1:
                            n = 1
                        elif dato[5] in num_senal2:
                            n = 2
                        elif dato[5] in num_senal3:
                            n = 3
                        elif dato[5] in num_senal4:
                            n = 4
                        elif dato[5] in num_senal5:
                            n = 5
                        elif dato[5] in num_senal6:
                            n = 6
                        if n != -1:
                            class_labels[n].append(n)
                            hog_vector = hog(imagen_recordata_escala)
                            class_features[n].append(hog_vector)
                        else:
                            encontrado = False
                            break
                        expanded_regions.append((new_x, new_y, new_w, new_h))

                    break

            if not encontrado:
                repetido = False
                for reg in expanded_regions:
                    if comparar_rectangulos(reg[0], reg[1], reg[2] + reg[0], reg[3] + reg[1], new_x,
                                            new_y, new_w + new_x, new_h + new_y):
                        repetido = True
                if not repetido:
                    hog_vector = hog(imagen_recordata_escala)
                    class_features[0].append(hog_vector)
                    class_labels[0].append(0)
    return expanded_regions


#Contraste y equalizacion de la imagen
def enhance_contrast(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)


#Cambiar el tamaño de las imagenes que conseguimos recortando
def resize_regions(image, target_size=(32, 32)):
    return cv2.resize(image, target_size)


def mser_func(original_image, min, max, datos):
    gray_image = enhance_contrast(original_image)
    mser = cv2.MSER_create(delta=3, min_area=min, max_area=max)
    regions, _ = mser.detectRegions(gray_image)
    expanded_regions = expand_detected_regions(regions, gray_image, original_image, datos)
    return expanded_regions

def KNN_learn(X_val, Y_val):
    knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors es el número de vecinos más cercanos

    # 3. Entrenar el clasificador KNN
    return knn.fit(X_val, Y_val)


def clasificador_binario():
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    X_total = [[]]
    y_total = [[]]

    for l,f in zip(class_labels,class_features):
        if ejercicio:
            X_train_aux, X_val_aux, y_train_aux, y_val_aux = train_test_split(f, l, test_size=0.2)

            X_train.append(X_train_aux)
            X_val.append(X_val_aux)

            y_train.append(y_train_aux)
            y_val.append(y_val_aux)
        else:
            X_train.append(f)
            y_train.append(l)
    #X_final = apply_LDA(X_train_lda[0], y_train_lda[0])

    for h, feature in enumerate(class_features[1:], 1):
        x = X_train[h]
        x.extend(X_train[0])
        feature_matrix = np.array(x)

        y = y_train[h]
        y.extend(y_train[0])
        class_matrix = np.array(y)

        #X_apply_LDA =apply_LDA(feature_matrix, class_matrix)
        if ejercicio:
            gnb_func(feature_matrix, class_matrix)
            x_vl = X_val[h]
            x_vl.extend(X_val[0])

            y_vl = y_val[h]
            y_vl.extend(y_val[0])
            y_total[0].extend(y_vl)

            #x_vl_with_lda =apply_LDA(x_vl, y_vl)

            X_total[0].extend(x_vl)

            y_pred = unique_classifier(x_vl, gnbs[-1])
            print("Clasificador binario ", h)

            mostrarMatriz(y_vl, y_pred)
        else:
            gnb_func(feature_matrix, class_matrix)

    if ejercicio:
        print("-------------------------------------------------------------------------------------------------")
        print("Clasificador multiclase formado por binarios ")
        y_pred, _ = np.array(multiclass_classifier(X_total[0]))
        mostrarMatriz(y_total[0], y_pred)


def clasificados_KNN():
    X_val_all = []
    Y_val_all = []
    for labels, feature in zip(class_labels, class_features):
        X_val_all.extend(feature)
        Y_val_all.extend(labels)
    #X_lda = apply_LDA(X_val_all,Y_val_all)
    feature_matrix = np.array(X_val_all)
    class_matrix = np.array(Y_val_all)
    X_train, X_val, y_train, y_val = train_test_split(feature_matrix, class_matrix, test_size=0.2)
    knn = KNN_learn(X_train, y_train)
    y_pred_train = knn.predict(X_val)
    conf_matrix_train = confusion_matrix(y_val, y_pred_train)
    print("Matriz de Confusión para el clasificador KNN:\n", conf_matrix_train)
    print("\nReporte de Clasificación (Datos de Entrenamiento):\n", classification_report(y_val, y_pred_train))
    return X_val, y_val


def apply_mser(image_paths, gt_txt):
    datos = [linea.strip().split(';') for linea in open(gt_txt, 'r')]
    for image_path in image_paths[:600]:
        #print(image_path)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"No se pudo cargar la imagen desde {image_path}")
            return

        datos_imagen = [arr for arr in datos if arr[0] == image_path[-9:]]
        expanded_regions = mser_func(original_image, 200, 20000, datos_imagen)

        #dibujar los cuadrados en la imagen
        for x, y, w, h in expanded_regions:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    if ejercicio:

        X_val, y_val = clasificados_KNN()

    clasificador_binario()
    if ejercicio:
        print("_------------------------------------------")
        print("claisificador binario con datos de knn")
        y_pred, _ = np.array(multiclass_classifier(X_val))
        mostrarMatriz(y_val, y_pred)

#interseccion over union
def comparar_rectangulos(x11, y11, x12, y12, x21, y21, x22, y22):
    # Coordenadas del rectángulo de intersección
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    # Calcular la superficie de la intersección
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou > 0.3

#ejercicio3
def apply_mser_from_test(image_paths):
    nombre_archivo = "resultado.txt"
    with open(nombre_archivo, 'w') as archivo:
        pass
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"No se pudo cargar la imagen desde {image_path}")
            return

        gray_image = enhance_contrast(original_image)
        regis = mser_detect(gray_image, 200, 10000)

        regs, hogs = expand_detected_regions_p1(regis, gray_image, original_image)

        if len(hogs) > 0:
            prediccion, porcentaje = multiclass_classifier(np.array(hogs))
            for pred, porc, r in zip(prediccion, porcentaje, regs):
                if pred != 0:
                    #print(image_path, pred, porc, r)
                    with open(nombre_archivo, 'a') as archivo:
                        archivo.write(
                            f"{image_path[-9:]};{r[0]};{r[1]};{r[0] + r[2]};{r[1] + r[3]};{pred};{porc}\n")


def mser_detect(gray, mini, maxi):
    mser = cv2.MSER_create(delta=3, min_area=mini, max_area=maxi)
    regions, _ = mser.detectRegions(gray)
    return regions

def expand_detected_regions_p1(regions, gray_image, original_image, expand_factor=1.2):
    expanded_regions = []
    hog_regions = []
    for region in regions:

        if len(region) == 4:
            x, y, w, h = region
        else:
            x, y, w, h = cv2.boundingRect(region)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            new_x = max(0, int(x - (expand_factor - 1) / 2 * w))
            new_y = max(0, int(y - (expand_factor - 1) / 2 * h))
            new_w = min(gray_image.shape[1], int(w * expand_factor))
            new_h = min(gray_image.shape[0], int(h * expand_factor))
            imagen_recortada = original_image[new_y:new_y + new_h, new_x:new_x + new_w]
            img = resize_regions(imagen_recortada)
            encontrado = False
            for reg in expanded_regions:
                if comparar_rectangulos(reg[0], reg[1], reg[2] + reg[0], reg[3] + reg[1], new_x,
                                        new_y, new_w + new_x, new_h + new_y):
                    encontrado = True
            if not encontrado:
                expanded_regions.append([new_x, new_y, new_w, new_h])
                hog_regions.append(hog(img))
    return expanded_regions, hog_regions

def mostrarMatriz(y_val, y_pred):
    # Calcular la matriz de confusión y otras métricas de rendimiento
    conf_matrix = confusion_matrix(y_val, y_pred)
    classification_rep = classification_report(y_val, y_pred)
    print("Matriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(classification_rep)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        apply_mser(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
