# importar librerias necesarias
import time

class PID:
	def __init__(self, kP=1, kI=0, kD=0):	# Constructor
		# inicializar las ganancias
		self.kP = kP
		self.kI = kI
		self.kD = kD

	def initialize(self):
		# inicializar el tiempo actual y el anterior
		self.currTime = time.time()
		self.prevTime = self.currTime

		# inicializar el error anterior
		self.prevError = 0

		# inicializar las variables de resultado de los términos
		self.cP = 0
		self.cI = 0
		self.cD = 0

	# Metodo donde se realizan los calculos 
	def update(self, error, sleep=0.2): #Valor de sleep debe tomar en cuenta limitaciones mecanicas y computacionales
		# pausa por un momento
		time.sleep(sleep)

		# toma la hora actual y calcula el tiempo delta
		self.currTime = time.time()
		deltaTime = self.currTime - self.prevTime

		# error delta
		deltaError = error - self.prevError

		# termino proporcional
		self.cP = error

		# termino proporcional
		self.cI += error * deltaTime

		# termino derivado y evitar la division por cero
		self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

		# guardar el tiempo y el error anteriores para la siguiente actualizacion
		self.prevTime = self.currTime
		self.prevError = error

		# suma las condiciones y devuelve
		return sum([
			self.kP * self.cP,
			self.kI * self.cI,
			self.kD * self.cD])
