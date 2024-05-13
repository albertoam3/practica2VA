import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer


num_senal1 = ['0', '1', '2', '3', '4', '5', '7', '8', '9', '10', '15', '16']
num_senal2 = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
num_senal3 = ['14']
num_senal4 = ['17']
num_senal5 = ['13']
num_senal6 = ['38']

senal1 = []
senal2 = []
senal3 = []
senal4 = []
senal5 = []
senal6 = []
nosenal = []

class_features = [[], [], [], [], [], [], []]
class_labels = [[], [], [], [], [], [], []]

gnbs =[]



def create_class_features():
    for i, hog_features in enumerate(class_features[1:], 1):
        hog_features.extend(class_features[0])
        class_labels[i].extend(class_labels[0])


def gnb_func(X_val,Y_val):

    X_train_lda=apply_LDA(X_val,Y_val)
    # Inicializar y ajustar el clasificador Bayesiano con Gaussianas
    gnb = GaussianNB()

    gnb.fit(X_train_lda, Y_val)
    gnbs.append(gnb)

def apply_LDA(X_val, y_val):
    #for i, hog_features in enumerate(class_features[1:], 1):
        hog_matrix = np.array(X_val)

        lda = LDA()
        lda.fit(hog_matrix, y_val)
        # Reducir la dimensionalidad de los datos de entrenamiento con LDA
        return lda.transform(hog_matrix)






def multiclass_classifier(X_val,y_val):

    X_train_lda = apply_LDA(X_val, y_val)

    # Obtener las probabilidades de pertenecer a cada clase para cada clasificador binario
    probabilities = []


    for gnb in gnbs:
        probabilities.append(gnb.predict_proba(X_train_lda))

    resultado = []

    for j,prob in enumerate(probabilities):
        #predicted_classes.append(np.argmax(prob, axis=1))
        pred = np.argmax(prob, axis=1)
        indices_y_probabilidades = list(zip(pred, np.max(prob, axis=1)))
        print(indices_y_probabilidades)
        for i, (indice, probabilidad) in enumerate(indices_y_probabilidades):

            if len(resultado)>i and probabilidad>resultado[i][1]:
                if indice == 0:
                    resultado[i] = [indice, probabilidad]
                else:
                    resultado[i] = [j+1, probabilidad]
            elif len(resultado)<=i:
                if indice == 0:
                    resultado.append([indice, probabilidad])
                else:
                    resultado.append([j + 1, probabilidad])


    resultado_final=[]
    for i,p in resultado:
        resultado_final.append(i)

    print(resultado_final)
    maximos_por_columna = np.amax(resultado_final, axis=0)
    return resultado_final

def hog(image, number):
    win_size = (32, 32)
    block_size = (4, 4)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_vector = hog.compute(image)
    class_features[number].append(hog_vector)
    class_labels[number].append(number)
    if(number !=0):
        print(number)
        print(hog_vector)


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
                        if dato[5] in num_senal1:
                            senal1.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 1)
                        elif dato[5] in num_senal2:
                            senal2.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 2)
                        elif dato[5] in num_senal3:
                            senal3.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 3)
                        elif dato[5] in num_senal4:
                            senal4.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 4)
                        elif dato[5] in num_senal5:
                            senal5.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 5)
                        elif dato[5] in num_senal6:
                            senal6.append(imagen_recordata_escala)
                            hog(imagen_recordata_escala, 6)
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
                    nosenal.append(imagen_recordata_escala)
                    hog(imagen_recordata_escala, 0)
                #expanded_regions.append((new_x, new_y, new_w, new_h))
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


def apply_mser(image_paths, gt_txt):

    datos = [linea.strip().split(';') for linea in open(gt_txt, 'r')]
    for image_path in image_paths[:100]:
        print(image_path)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"No se pudo cargar la imagen desde {image_path}")
            return

        datos_imagen = [arr for arr in datos if arr[0] == image_path[-9:]]

        expanded_regions = mser_func(original_image, 200, 20000, datos_imagen)

        #dibujar los cuadrados en la imagen
        for x, y, w, h in expanded_regions:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #cv2.imshow("original", original_image)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    create_class_features()

    X_val_all = []
    y_val_all = []

    for h,feature in enumerate(class_features):
        if h==1 or h==5:
            feature_matrix = np.array(feature)
            class_matrix = np.array(class_labels[h])

            X_train, X_val, y_train, y_val = train_test_split(feature_matrix, class_matrix, test_size=0.2)
            X_val_all.append(X_val)
            y_val_all.append(y_val)
    # Aplicar LDA y entrenar clasificadores binarios
            gnb_func(X_train, y_train)



    for X_val, y_val in zip(X_val_all, y_val_all):
        y_pred = np.array(multiclass_classifier(X_val,y_val))
        # Calcular la matriz de confusión y otras métricas de rendimiento
        conf_matrix = confusion_matrix(y_val, y_pred)
        classification_rep = classification_report(y_val, y_pred)

        print("Matriz de Confusión:")
        print(conf_matrix)
        print("\nReporte de Clasificación:")
        print(classification_rep)



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

    if iou > 0.4:

        return True
    else:
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:

        apply_mser(sys.argv[1])
    else:
        print("Usage: python detector.py path_to_image")
