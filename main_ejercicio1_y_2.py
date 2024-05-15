import argparse
import os

import cv2

import Detector_antiguo
import Detector

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Nombre del detector a ejecutar')
    parser.add_argument(
        '--train_path', default="", help='Carpeta con las imágenes de entrenamiento')

    args = parser.parse_args()

    train_path = args.train_path




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


    Detector.ejercicio_check(True)
    Detector.apply_mser(image_paths,gt_txt)
    # Create the detector
    print("Creando el detector " + args.detector)




    # Guardar resultados en el fichero resultado.txt

    # Guardar resultados en el fichero resultado_por_tipo.txt






