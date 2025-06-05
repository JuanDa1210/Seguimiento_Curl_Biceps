import cv2
import mediapipe as mp
import numpy as np
import time
import customtkinter as ctk
import os 

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a) # Hombro
    b = np.array(b) # Codo
    c = np.array(c) # Muñeca

   
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

   
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)


    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def elegir_brazo_gui():
    
    root = ctk.CTk()
    root.geometry("400x200")
    root.title("Seleccion del Brazo")

    eleccion = None 

    def seleccionar(brazo_seleccionado):
        nonlocal eleccion 
        eleccion = brazo_seleccionado
        root.quit() 
        root.destroy() 
    label = ctk.CTkLabel(root, text="¿Qué brazo deseas usar para el ejercicio?", font=("Arial", 16))
    label.pack(pady=20)

    boton_derecho = ctk.CTkButton(root, text="Brazo Derecho", command=lambda: seleccionar("Derecho"), width=200)
    boton_derecho.pack(pady=10)

    boton_izquierdo = ctk.CTkButton(root, text="Brazo Izquierdo", command=lambda: seleccionar("Izquierdo"), width=200)
    boton_izquierdo.pack(pady=10)

    root.mainloop() 

    return eleccion 

brazo_elegido = elegir_brazo_gui()

# Asignar landmarks según la elección
if brazo_elegido == "Izquierdo":
    shoulder_landmark = mp_pose.PoseLandmark.LEFT_SHOULDER
    elbow_landmark = mp_pose.PoseLandmark.LEFT_ELBOW
    wrist_landmark = mp_pose.PoseLandmark.LEFT_WRIST
    nombre_archivo_csv = "Prueba1_izq.csv" # Nombre de archivo consistente
else: # Por defecto o si es "derecho"
    shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER
    elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW
    wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST
    nombre_archivo_csv = "Prueba1_der.csv" # Nombre de archivo consistente

# Inicializa la captura de video
cap = cv2.VideoCapture(0)

try:

    with open(nombre_archivo_csv, "w") as archivo_salida:
        archivo_salida.write("Tiempo,Angulo\n") 

    archivo_salida = open(nombre_archivo_csv, "a")

except IOError as e:
    print(f"Error al abrir el archivo CSV '{nombre_archivo_csv}': {e}")
    cap.release()
    exit()


# Iniciar MediaPipe Pose
# Iniciar MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    start_time = None  # Inicia en None
    duracion_captura = 10

    while True:  
        ret, frame = cap.read()
        if not ret:
            print("Error al leer fotograma de la cámara.")
            break

        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            if start_time is None:
                print("Usuario detectado, iniciando conteo de tiempo.")
                start_time = time.time()

            elapsed_time = time.time() - start_time
            if elapsed_time > duracion_captura:
                print("Tiempo de captura finalizado.")
                break

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[shoulder_landmark.value].x, landmarks[shoulder_landmark.value].y]
                elbow = [landmarks[elbow_landmark.value].x, landmarks[elbow_landmark.value].y]
                wrist = [landmarks[wrist_landmark.value].x, landmarks[wrist_landmark.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                archivo_salida.write(f"{elapsed_time:.4f},{angle:.2f}\n")

                shoulder_coords = tuple(np.multiply(shoulder, [w, h]).astype(int))
                elbow_coords = tuple(np.multiply(elbow, [w, h]).astype(int))
                wrist_coords = tuple(np.multiply(wrist, [w, h]).astype(int))

                cv2.line(image_bgr, shoulder_coords, elbow_coords, (0, 255, 0), 3)
                cv2.line(image_bgr, elbow_coords, wrist_coords, (0, 255, 0), 3)
                cv2.circle(image_bgr, shoulder_coords, 6, (0, 0, 255), -1)
                cv2.circle(image_bgr, elbow_coords, 6, (0, 0, 255), -1)
                cv2.circle(image_bgr, wrist_coords, 6, (0, 0, 255), -1)


                cv2.putText(image_bgr, f"{angle:.1f}",
                            (elbow_coords[0] - 40, elbow_coords[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Tiempo: {elapsed_time:.1f}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                
            except Exception as e:
                pass

        else:
            if start_time is None:
                cv2.putText(image_bgr, "Esperando Usuario", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Deteccion de Angulo', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

archivo_salida.close()
cap.release()
cv2.destroyAllWindows()
print("Captura finalizada y archivo guardado.")