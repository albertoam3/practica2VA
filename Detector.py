import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

gnbs = [[], [], [], [], [], []]

def apply_LDA():
    i=1
    for hog_features in class_features[1:]:
        hog_features.extend(class_features[0])
        class_labels[i].extend(class_labels[0])
        i=i+1
    i=1
    for hog_features in class_features[1:]:
        hog_matrix = np.array(hog_features)
        n_muestras, n_features = hog_matrix.shape

        lda = LDA()
        lda.fit(hog_matrix, class_labels[i])


         # Reducir la dimensionalidad de los datos de entrenamiento con LDA
        X_train_lda = lda.transform(hog_matrix)

        # Inicializar y ajustar el clasificador Bayesiano con Gaussianas
        gnbs[i-1] = GaussianNB()
        gnbs[i-1].fit(X_train_lda, class_labels[i])

        #dan datos que hay que borrar
        n_classes = len(np.unique(class_labels[i]))
        print("Número de características:", n_features)
        print("Número de clases:", n_classes)
        i=i+1

def multiclass_classifier(sample):
    # Obtener las probabilidades de pertenecer a cada clase para cada clasificador binario
    probabilities = []
    for gnb in gnbs:
        probabilities.append(gnb.predict_proba(sample.reshape(1, -1))[0])

    # Obtener la clase con la mayor probabilidad para cada clasificador binario
    predicted_classes = [np.argmax(prob) for prob in probabilities]

    # Obtener las etiquetas de clase correspondientes
    predicted_labels = LabelBinarizer.inverse_transform(np.array(predicted_classes).reshape(1, -1))

    return predicted_labels[0]

def hog(image, number):
    win_size = (32, 32)
    block_size = (8, 8)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_vector = hog.compute(image)
    class_features[number].append(hog_vector)
    class_labels[number].append(number)


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

#    plt.imshow(gray_image, cmap='gray')
#    plt.title('Imagen en escala de grises')
#    plt.show()

    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
#    plt.plot(hist_original)
#    plt.title('Histograma de la imagen en escala de grises')
#    plt.show()

    equ_image = cv2.equalizeHist(gray_image)

#    plt.imshow(equ_image, cmap='gray')
#    plt.title('Imagen con contraste mejorado')
#    plt.show()

    hist_equ = cv2.calcHist([equ_image], [0], None, [256], [0, 256])
#    plt.plot(hist_equ)
#    plt.title('Histograma de la imagen con contraste mejorado')
#    plt.show()

    return equ_image


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
    print(image_paths)

    datos = [linea.strip().split(';') for linea in open(gt_txt, 'r')]
    for image_path in image_paths:
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
    apply_LDA()


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
