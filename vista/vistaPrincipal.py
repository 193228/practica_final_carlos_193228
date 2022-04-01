# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'vistaPrincipal.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1308, 858)
        MainWindow.setStyleSheet("background-color: rgb(66, 133, 244);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 60, 441, 61))
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.seleccionCarpeta = QtWidgets.QPushButton(self.centralwidget)
        self.seleccionCarpeta.setGeometry(QtCore.QRect(190, 160, 161, 41))
        self.seleccionCarpeta.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.seleccionCarpeta.setObjectName("seleccionCarpeta")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(750, 60, 331, 61))
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.seleccionImagen = QtWidgets.QPushButton(self.centralwidget)
        self.seleccionImagen.setGeometry(QtCore.QRect(840, 160, 161, 41))
        self.seleccionImagen.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.seleccionImagen.setObjectName("seleccionImagen")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 380, 341, 61))
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(510, 380, 291, 61))
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.radioTaza = QtWidgets.QRadioButton(self.centralwidget)
        self.radioTaza.setGeometry(QtCore.QRect(570, 540, 161, 20))
        self.radioTaza.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.radioTaza.setCheckable(True)
        self.radioTaza.setChecked(True)
        self.radioTaza.setAutoExclusive(False)
        self.radioTaza.setObjectName("radioTaza")
        self.radioEpocas = QtWidgets.QRadioButton(self.centralwidget)
        self.radioEpocas.setGeometry(QtCore.QRect(160, 540, 171, 20))
        self.radioEpocas.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.radioEpocas.setCheckable(True)
        self.radioEpocas.setChecked(True)
        self.radioEpocas.setAutoExclusive(False)
        self.radioEpocas.setObjectName("radioEpocas")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(920, 380, 181, 61))
        self.label_5.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.label_5.setObjectName("label_5")
        self.radioLote = QtWidgets.QRadioButton(self.centralwidget)
        self.radioLote.setGeometry(QtCore.QRect(930, 540, 161, 20))
        self.radioLote.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"")
        self.radioLote.setCheckable(True)
        self.radioLote.setChecked(True)
        self.radioLote.setAutoExclusive(False)
        self.radioLote.setObjectName("radioLote")
        self.epocas = QtWidgets.QLineEdit(self.centralwidget)
        self.epocas.setGeometry(QtCore.QRect(120, 470, 231, 31))
        self.epocas.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.epocas.setText("")
        self.epocas.setObjectName("epocas")
        self.tazaAprendizaje = QtWidgets.QLineEdit(self.centralwidget)
        self.tazaAprendizaje.setGeometry(QtCore.QRect(530, 460, 231, 31))
        self.tazaAprendizaje.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.tazaAprendizaje.setText("")
        self.tazaAprendizaje.setObjectName("tazaAprendizaje")
        self.lote = QtWidgets.QLineEdit(self.centralwidget)
        self.lote.setGeometry(QtCore.QRect(890, 460, 231, 31))
        self.lote.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.lote.setText("")
        self.lote.setObjectName("lote")
        self.botonEjecucion = QtWidgets.QPushButton(self.centralwidget)
        self.botonEjecucion.setGeometry(QtCore.QRect(570, 690, 161, 41))
        self.botonEjecucion.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 12pt \"MS Shell Dlg 2\";")
        self.botonEjecucion.setObjectName("botonEjecucion")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1308, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Selecciona la carpeta del dataset a analizar"))
        self.seleccionCarpeta.setText(_translate("MainWindow", "Subir Carpeta"))
        self.label_2.setText(_translate("MainWindow", "Selecciona la imagen a predecir"))
        self.seleccionImagen.setText(_translate("MainWindow", "Analizar"))
        self.label_3.setText(_translate("MainWindow", "Ingrese numero de generaciones"))
        self.label_4.setText(_translate("MainWindow", "Ingrese taza de aprendizaje"))
        self.radioTaza.setText(_translate("MainWindow", "Mejor Taza (0.01)"))
        self.radioEpocas.setText(_translate("MainWindow", "Recomendado (50)"))
        self.label_5.setText(_translate("MainWindow", "Tamaño De Lote"))
        self.radioLote.setText(_translate("MainWindow", "4 (Opcional)"))
        self.botonEjecucion.setText(_translate("MainWindow", "Ejecucion"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
