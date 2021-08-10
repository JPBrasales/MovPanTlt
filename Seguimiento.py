#importar librerias necesarias
from multiprocessing import Manager
from multiprocessing import Process
from Inferencias import *
from Control import PID
import signal
import time
import sys
import cv2
import pigpio
import pandas as pd

# definir el rango seguro de los motores
servoRangePan = (500,2490)
servoRangeTlt = (1670, 2490)

# funcion para interrunpir la ejecucion con CTRL+C
def signal_handler(sig, frame):
    sys.exit()

def obj_center(objX,objY,centerX,centerY):
    signal.signal(signal.SIG_IGN,signal_handler)
    ObjDetector().start(objX,objY,centerX,centerY)


def pid_pross(output, p, i, d, objCoord, centerCoord):
    signal.signal(signal.SIGINT, signal_handler)
    # crear el PID e inicializarlo
    p = PID(p.value, i.value, d.value)
    p.initialize()
    contador = 0

    # loop indefinido
    while True:
        # calcular el error
        error = centerCoord.value - objCoord.value

        #Guardar los datos en un registro
        contador = contador + 1
        df = open('panCE2.csv','a')
        df.write(str(objCoord.value))
        df.write('\n')
        
        # actualizamos el valor de salida 
        if objCoord.value == 0  or error ==480 or error ==270:
            output.value = 0
        else:
            output.value = p.update(error)
        #print(error,output)

def in_range(val, start, end):
    return val >= start and val <= end
        
def set_servos(pan_delta, tilt_delta):
    #Declaramos pines y posicion inicial
    signal.signal(signal.SIGINT, signal_handler)
    pan = 27
    tlt = 17
    pi = pigpio.pi()
    pi.set_mode(pan, pigpio.OUTPUT)
    pi.set_mode(tlt, pigpio.OUTPUT)
    pi.set_servo_pulsewidth(pan, 800)
    time.sleep(0.3)
    pi.set_servo_pulsewidth(tlt, 1790)
    
    while True:
        #Calcular movimientos
        pan_change = pan_delta.value * -1
        pan_pulse_width =  pi.get_servo_pulsewidth(pan) + pan_change
        tlt_change = tilt_delta.value * -1
        tlt_pulse_width =  pi.get_servo_pulsewidth(tlt) + tlt_change
        
        #comprobar dentro del rango y movimiento
        if in_range(pan_pulse_width, servoRangePan[0], servoRangePan[1]):
            pi.set_servo_pulsewidth(pan, pan_pulse_width)
            #print(pan_pulse_width)
        if in_range(tlt_pulse_width, servoRangeTlt[0], servoRangeTlt[1]):
            pi.set_servo_pulsewidth(tlt, tlt_pulse_width)
            #print(tlt_pulse_width)

        time.sleep(0.05)

if __name__ == "__main__":

    # iniciar el gestor para las variables e iniciar los procesos
    with Manager() as manager:

        # valores enteros para las coordenadas del centro del frame (x, y)
        centerX = manager.Value("i", 0)
        centerY = manager.Value("i", 0)

        # valores enteros para las coordenadas (x, y) del objeto
        objX = manager.Value("i", 0)
        objY = manager.Value("i", 0)

        # valores de giro e inclinación serán gestionados por PIDs independientes
        pan_delta = manager.Value("i", 0)
        tlt_delta = manager.Value("i", 0)

        # establecer los valores PID para el desplazamiento panoramico
        panP = manager.Value("f", 0.062)
        panI = manager.Value("f", 0.0055)
        panD = manager.Value("f", 0.0022)

        # establecer valores PID para la inclinación
        tiltP = manager.Value("f", 0.07)
        tiltI = manager.Value("f", 0.0045)
        tiltD = manager.Value("f", 0.0022)

        #4 procesos independientes
        prossObjCenter = Process(target=obj_center, args=(objX,objY,centerX,centerY)) #localiza los objetivos
        prossPan = Process(target=pid_pross, args=(pan_delta, panP, panI, panD, objX, centerX)) #determina el valor de giro
        prossTilt = Process(target=pid_pross, args=(tlt_delta, tiltP, tiltI, tiltD, objY, centerY)) #determina el valor de inclinación
        prossSetServos = Process(target=set_servos, args=(pan_delta, tlt_delta)) #acción de motores

        # iniciamos los 4 procesos
        prossObjCenter.start()
        prossPan.start()
        prossTilt.start()
        prossSetServos.start()

        # Unimos los 4 procesos
        prossObjCenter.join()
        prossPan.join()
        prossTilt.join()
        prossSetServos.join()

