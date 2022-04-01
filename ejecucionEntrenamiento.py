import PyQt5
from matplotlib import pyplot as plt
from ajuste import ajusteDatos
from controlador.redNeuronal import *
from modelado.modelo import modelo
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

paths = []

def ejecucionAlgoritmo(ventana):
    print(paths)
    # obtenemos los parametros a necesitar
    datos = ajusteDatos(ventana)
    # procesamos y reducimos las imagenes
    data, labels = procesadoImagenes(paths[0])
    # dividimos nuestros datos
    trainX, testX, trainY, testY = divisionDatos(data,labels)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    # construir el generador de imágenes para el aumento de datos
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    # inicializamos nuestra red neuronal convolucional
    model = modelo.build(width=64, height=64, depth=3, classes=len(lb.classes_))
    # entrenamos nuestra red
    entrenamiento = entrenamientoRed(datos,model,aug,trainX,trainY,testX,testY)
    # evaluamos nuestro red
    evaluarRed(model,testX,testY,lb)
    #graficamos la perdida
    graficacionPerdida(entrenamiento,datos['epocas'])
    #graficamos la precision
    graficacionPrecision(entrenamiento,datos['epocas'])
    try:
        graficaPrediccion(paths[1], model, lb)  # si no carga la  imagen no es hace
    except:
        pass
    #resumen del modelo
    model.summary()

def carpetaAnalizar(ventana):
    folder = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(None,"Seleccione Directorio Imagenes")
    return paths.append(folder)

def obtencionImagen(ventana):
    if len(paths)!= 0:
        file = PyQt5.QtWidgets.QFileDialog.getOpenFileName(None, "Seleccione Imagen", "", "Image Files(*.png *.jpg *.bmp)")
        return paths.append(file[0])
    else:
        return PyQt5.QtWidgets.QMessageBox.about(None, "No ah cargado dataset", "Cargue la carpeta del dataset antes de analizar")

def graficacionPerdida(entrenamiento,epocas):
    # plot
    N = np.arange(0, epocas)
    plt.style.use("ggplot")
    plt.figure(figsize=[8, 6])
    plt.plot(N, entrenamiento.history["loss"], label="perdida de entrenamiento")
    plt.plot(N, entrenamiento.history["val_loss"], label="valor de perdida")
    plt.title("Perdida Red Neuronal Convolucional")
    plt.xlabel("Numero de epoca", weight='bold')
    plt.ylabel("Perdida", weight='bold')
    plt.legend()
    plt.show()
    plt.savefig("Resultados/grafica de perdida.png")

def graficacionPrecision(entrenamiento,epocas):
    # plot
    N = np.arange(0, epocas)
    plt.style.use("ggplot")
    plt.figure(figsize=[8, 6])
    plt.plot(N, entrenamiento.history["accuracy"], label="precision de entrenamiento")
    plt.plot(N, entrenamiento.history["val_accuracy"], label="valor de precision")
    plt.title("Precision Red Neuronal Convolucional")
    plt.xlabel("Numero de epoca", weight='bold')
    plt.ylabel("presicion", weight='bold')
    plt.legend()
    plt.show()
    plt.savefig("Resultados/grafica de presicion.png")

def graficaPrediccion(imagen,model,lb):
    # cargamos las imagenes
    width = 64
    height = 64
    image = cv2.imread(imagen)
    output = image.copy()
    image = cv2.resize(image, (width, height))
    image = image.astype("float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    text = "{}: {:.1f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    print(output)
    # mostramos la salida de la imagen
    cv2.imshow("Image", output)
    cv2.waitKey(0)  # Delay in milliseconds

    plt.style.use("ggplot")
    plt.figure(figsize=[10, 5])  # [width, height]

    x = [lb.classes_[0], lb.classes_[1], lb.classes_[2], lb.classes_[3], lb.classes_[4]]
    y = [preds[0][0], preds[0][1], preds[0][2], preds[0][3], preds[0][4]]
    plt.barh(x, y, color='violet')

    ticks_x = np.linspace(0, 1, 11)
    plt.xticks(ticks_x, fontsize=10, family='fantasy', color='black')
    plt.yticks(size=15, color='navy')
    for i, v in enumerate(y):
        plt.text(v, i, "  " + str((v * 100).round(1)) + "%", color='blue', va='center', fontweight=None)

    plt.title('Prediction Probability', family='serif', fontsize=15, style='italic', weight='bold', color='olive',loc='center', rotation=0)
    plt.xlabel('Probability', fontsize=12, weight='bold', color='blue')
    plt.ylabel('Category', fontsize=12, weight='bold', color='indigo')
    plt.show()
    plt.savefig("Resultados/Probabilidad De Prediccion.png")