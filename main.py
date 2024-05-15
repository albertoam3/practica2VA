import argparse
import os

import cv2

import Detector2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Nombre del detector a ejecutar')
    parser.add_argument(
        '--train_path', default="", help='Carpeta con las imágenes de entrenamiento')
    parser.add_argument(
        '--test_path', default="", help='Carpeta con las imágenes de test')

    args = parser.parse_args()

    train_path = args.train_path

    test_path = args.test_path


    # Cargar los datos de entrenamiento sin se necesita
    print("Cargando datos de entrenamiento desde " + args.train_path)



    image_paths = []
    print(train_path)
    # Obtener la lista de archivos en la carpeta train_path
    gt_txt=''
    i=0
    for filename in os.listdir(train_path):
        if filename.endswith(".ppm"):  # Filtrar solo archivos de imagen
            image_paths.append(os.path.join(train_path, filename))
        else:
            gt_txt=train_path+'/'+filename



    Detector2.ejecicio_check(False)
    Detector2.apply_mser(image_paths,gt_txt)


    image_test_path = []
    for filename in os.listdir(test_path+"_alumnos"):
        if filename.endswith(".ppm"):  # Filtrar solo archivos de imagen
            image_test_path.append(os.path.join(test_path+"_alumnos", filename))

    Detector2.apply_mser_from_test(image_test_path)
    # Create the detector
    print("Creando el detector " + args.detector)

    # Cargar los datos de test y ejecutar el detector en esas imágenes
    print("Probando el detector " + args.detector + " en " + args.test_path)



    # Guardar resultados en el fichero resultado.txt

    # Guardar resultados en el fichero resultado_por_tipo.txt






