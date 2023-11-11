from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

# Charger un modèle pré-entraîné
model = ResNet50(weights='imagenet')

# Fonction pour prédire si l'image contient un chien
def dog_detector(img):
    img = cv2.resize(img, (224, 224))  # Redimensionner l'image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir l'image en RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prediction = model.predict(img)
    return (np.argmax(prediction) <= 268) and (np.argmax(prediction) >= 151)

# Fonction pour la détection en temps réel
def live_dog_detection(camera, label):
    # Access the camera texture and convert it to a numpy array
    frame = np.frombuffer(camera.texture.pixels, dtype='uint8')
    frame = frame.reshape((camera.texture.height, camera.texture.width, 4))

    # Détection de chien en temps réel
    if dog_detector(frame):
        text = "Chien détecté"
    else:
        text = "Pas de chien détecté"

    label.text = text

KV = '''
BoxLayout:
    orientation: 'vertical'

    Camera:
        id: camera
        resolution: (640, 480)
        play: True

    MDLabel:
        id: detection_label
        text: "Attente de détection..."
        halign: 'center'
'''

class MonApplication(MDApp):

    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        camera = self.root.ids.camera
        label = self.root.ids.detection_label
        # Use the 'on_texture' event to trigger the live_dog_detection function
        camera.bind(on_texture=lambda instance: self.live_dog_detection(instance, label))

if __name__ == "__main__":
    MonApplication().run()
