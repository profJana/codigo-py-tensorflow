import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('keras_model.h5') #modelo que treinamos

video = cv2.VideoCapture(0) #captura da webcam

while True:

    check,frame = video.read() 

    # Altere o dado de entrada:

    # 1. Redimensione a imagem

    img = cv2.resize(frame,(224,224)) #redimensiona o quadro de video para a resolução em pixels
    #fazemos isso pois o tensorflow nos pede para que a imagem de entrada seja desse tamanho 224, 224



    # 2. Converta a imagem em um array Numpy e aumente a dimensão

    test_image = np.array(img, dtype=np.float32)
    #img é convertida em um array numpy com tipo float32 (numeros decimais)

    test_image = np.expand_dims(test_image, axis=0)
    #usado para adicionar uma dimensão extra no array, axis=0 add no inicio do array

    # 3. Normalize a imagem
    normalised_image = test_image/255.0
    #é pego os valores dos pixels da imagem e dividido pelo máximo de pixel (255)
    #isso ajuda a garantir que a entrada de dado esteja na faixa adequada da rede neural

    # Preveja o resultado
    prediction = model.predict(normalised_image)
    prediction_formatted = [f'{p:.2f}' for p in prediction[0]]
    print("Previsão: ", prediction_formatted)
        
    cv2.imshow("Resultado",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Fechando")
        break

video.release()