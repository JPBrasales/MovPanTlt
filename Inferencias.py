# Importando librerias necesarias
import os
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pyautogui

'''Definir la clase VideoStream para manejar la 
   transmisión de video en un hilo de procesamiento separado por Adrian Rosebrock'''

class VideoStream:
    def __init__(self,resolution,framerate):
        # Inicializar la PiCamera y el flujo de imágenes de la cámara
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Leer el primer fotograma del flujo
        (self.grabbed, self.frame) = self.stream.read()

	    # Variable para controlar cuándo se detiene la cámara
        self.stopped = False

    def start(self):
	    # Iniciar el hilo que lee los fotogramas del flujo de vídeo
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Mantener el bucle indefinidamente hasta que el hilo se detenga
        while True:
            # Si la cámara está detenida, detenga el hilo
            if self.stopped:
                # Cerrar los recursos de la cámara
                self.stream.release()
                return

            # En caso contrario, coge el siguiente fotograma del flujo
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # Devuelve el fotograma más reciente
        return self.frame

    def stop(self):
	    # Indicar que la cámara y el hilo deben detenerse
        self.stopped = True


class ObjDetector:
    def __init__(self):

        #Definimos variables y modelos
        MODEL_NAME = 'TFLite_model'
        GRAPH_NAME = 'model_edgetpu.tflite'
        LABELMAP_NAME = 'labelmap.txt'
        self.min_umbral = 0.7
        self.imW, self.imH = 960,540
        use_TPU = True
        pkg = importlib.util.find_spec('tflite_runtime')

        #Verificamos la instalación de la librería para uso de TPU
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
                print('Usando TPU')
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # Ruta del directorio de trabajo actual
        CWD_PATH = os.getcwd()

        # Ruta al archivo .tflite, que contiene el modelo
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Ruta de acceso al archivo de mapa de etiquetas
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Cargar el mapa de etiquetas
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        if use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

    def start(self,objX, objY,centerX,centerY):
        # Obtener detalles del modelo a ejecutar
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)

        #Valores para normalizar si fuese necesario
        input_mean = 127.5
        input_std = 127.5

        # Inicializar el cálculo de la velocidad de fotogramas
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        # Inicializar videoStream
        videostream = VideoStream(resolution=(self.imW,self.imH),framerate=30).start()
        time.sleep(1)

        while True:

            # Temporizador de inicio (para calcular la velocidad de fotogramas)
            t1 = cv2.getTickCount()

            # Tomar un fotograma del video stream
            frame1 = videostream.read()

            # Adquirir el marco y cambiar el tamaño
            frame = frame1.copy()
            frame = cv2.flip(frame,0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalizar los valores de los píxeles si se utiliza un modelo flotante (es decir, si el modelo no está cuantificado)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Realizar la detección real ejecutando el modelo con la imagen como entrada
            self.interpreter.set_tensor(input_details[0]['index'],input_data)
            self.interpreter.invoke()

            # Recuperar los resultados de la detección
            boxes = self.interpreter.get_tensor(output_details[0]['index'])[0] # Coordenadas de la caja delimitadora de los objetos detectados
            classes = self.interpreter.get_tensor(output_details[1]['index'])[0] # Índice de clase de los objetos detectados
            scores = self.interpreter.get_tensor(output_details[2]['index'])[0] # Confianza de los objetos detectados

            # Recorrer todas las detecciones y dibujar el cuadro de detección si la confianza está por encima del umbral mínimo
            for i in range(len(scores)):
                if ((scores[i] > self.min_umbral) and (scores[i] <= 1.0)):

                    # Obtener las coordenadas de la caja delimitadora y dibujarla dentro del marco

                    ymin = int(max(1,(boxes[i][0] * self.imH)))
                    xmin = int(max(1,(boxes[i][1] * self.imW)))
                    ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                    xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                    
                    objX.value = 0
                    objY.value = 0
                    
                    #Verificamos el objeto detectado
                    if self.labels[int(classes[i])] =="Mask":
                        
                        objX.value = (xmax + xmin)/2
                        objY.value = (ymax + ymin)/2
                        centerX.value=self.imW/2
                        centerY.value=self.imH/2

                        cv2.circle(frame, (int(objX.value),int(objY.value)), 5, (0, 0, 255), -1) #centro del objeto
                        cv2.circle(frame, (int(centerX.value),int(centerY.value)), 5, (255, 0, 0), -1) #Centro del marco
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Dibujar etiquetas
                    object_name = self.labels[int(classes[i])] # Busca el nombre del objeto en la matriz
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Obtener el tamaño de la fuente
                    label_ymin = max(ymin, labelSize[1] + 10) # se asegura de no salirse del marco
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Dibujar el texto de la etiqueta
                    time.sleep(0.001)
                    
            #  Calcular la velocidad de fotogramas
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # mostrar la velocidad de fotogramas
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            cv2.imshow('Object detector', frame)

            # Salir
            if cv2.waitKey(1) == ord('q'):
                print('Cerrando')
                break
                
        # Limpiar
        cv2.destroyAllWindows()
        #pyautogui.hotkey('ctrl', 'c') # Press the Ctrl-C hotkey combination.
        videostream.stop()
        