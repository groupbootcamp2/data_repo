from ctypes.wintypes import RGB

import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QRect, QSize, QPoint
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QGroupBox,
                             QHBoxLayout, QLabel, QMenu, QMenuBar, QPushButton,
                             QVBoxLayout, QFileDialog, QRubberBand, QTabWidget, QWidget, QGridLayout)
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import time


import compiled_model as cm
import create_data as cd
import data_visualization as dv
import model_visualization as mv
import config as cnfg
import gradcam



# Main window class
class Camera(QMainWindow):

    # constructor
    def __init__(self,parent_class):
        super().__init__()
        self.parent_class=parent_class
        # setting geometry
        self.setGeometry(100, 100, 800, 600)
        # setting style sheet
        self.setStyleSheet("background : lightgrey;")
        # getting available cameras
        self.available_cameras = QCameraInfo.availableCameras()
        # if no camera found
        if not self.available_cameras:
            sys.exit()
        # creating a status bar
        self.status = QStatusBar()
        # setting style sheet to the status bar
        self.status.setStyleSheet("background : white;")
        # adding status bar to the main window
        self.setStatusBar(self.status)
        # path to save
        self.save_path = r"C:\Users\User\Desktop\studing\bootcamp\AMAT\project\data_repo\photos"
        # creating a QCameraViewfinder object
        self.viewfinder = QCameraViewfinder()
        # showing this viewfinder
        self.viewfinder.show()
        # making it central widget of main window
        self.setCentralWidget(self.viewfinder)
        # Set the default camera.
        self.select_camera(0)
        # creating a tool bar
        toolbar = QToolBar("Camera Tool Bar")
        # adding tool bar to main window
        self.addToolBar(toolbar)
        # creating a photo action to take photo
        click_action = QAction("Click photo", self)
        # adding status tip to the photo action
        click_action.setStatusTip("This will capture picture")
        # adding tool tip
        click_action.setToolTip("Capture picture")
        # adding action to it
        # calling take_photo method
        click_action.triggered.connect(self.click_photo)
        # adding this to the tool bar
        toolbar.addAction(click_action)
        # similarly creating action for changing save folder
        change_folder_action = QAction("Change save location", self)
        # adding status tip
        change_folder_action.setStatusTip("Change folder where picture will be saved saved.")
        # adding tool tip to it
        change_folder_action.setToolTip("Change save location")
        # setting calling method to the change folder action
        # when triggered signal is emitted
        change_folder_action.triggered.connect(self.change_folder)
        # adding this to the tool bar
        toolbar.addAction(change_folder_action)
        # setting tool bar stylesheet
        toolbar.setStyleSheet("background : white;")
        # setting window title
        if self.parent_class==0:
            self.setWindowTitle("camera for predict the image")
        elif self.parent_class==1:
            self.setWindowTitle("camera for adding image to test")

        # showing the main window
        self.show()

    # method to select camera
    def select_camera(self, i):
        # getting the selected camera
        self.camera = QCamera(self.available_cameras[i])
        # setting view finder to the camera
        self.camera.setViewfinder(self.viewfinder)
        # setting capture mode to the camera
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        # if any error occur show the alert
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        # start the camera
        self.camera.start()
        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)
        # showing alert if error occur
        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg))
        # when image captured showing message
        self.capture.imageCaptured.connect(lambda d,  i: self.status.showMessage("Image captured : " + str(self.save_seq)))
        self.capture.imageSaved.connect(self.show_after_save)
        # getting current camera name
        self.current_camera_name = self.available_cameras[i].description()
        # initial save sequence
        self.save_seq = 0

    # method to take photo
    def click_photo(self):
          # time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        self.image_path = os.path.join(self.save_path,  "%04d-%s" % (self.save_seq, timestamp))
        # capture the image and save it on the save path
        self.capture.capture(self.image_path)
        # increment the sequence
        self.save_seq += 1

    def show_after_save(self):
        dialog.tabs_list[self.parent_class].showImage(self.image_path+".jpg")

    # change folder method
    def change_folder(self):
        # open the dialog to select path
        path = QFileDialog.getExistingDirectory(self, "Picture Location", "")
        # if path is selected
        if path:
            # update the path
            self.save_path = path
            # update the sequence
            self.save_seq = 0

    # method for alerts
    def alert(self, msg):
        # error message
        error = QErrorMessage(self)
        # setting text to the error message
        error.showMessage(msg)


class imageQLabel(QLabel):
    def __init__(self, parent_class=None):
        super().__init__()
        self.parent_class:int=parent_class

    def mousePressEvent(self, mouse_event: QMouseEvent):
        self.origin_point = mouse_event.pos()
        self.current_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.current_rubber_band.setGeometry(QRect(self.origin_point, QSize()))
        self.current_rubber_band.show()

    def mouseMoveEvent(self, mouse_event: QMouseEvent):
        self.current_rubber_band.setGeometry(QRect(self.origin_point, mouse_event.pos()).normalized())

    def mouseReleaseEvent(self, mouse_event: QMouseEvent):
        self.current_rubber_band.hide()
        current_rect: QRect = self.current_rubber_band.geometry()
        self.current_rubber_band.deleteLater()
        # current_rect.setRect(current_rect.getRect()[0]-((self.width()-self.pixmap().width())//2),current_rect.getRect()[1]-((self.height()-self.pixmap().height())//2),current_rect.getRect()[2],current_rect.getRect()[3])
        crop_pixmap: QPixmap = self.pixmap().copy(current_rect)
        crop_pixmap.save(f"output{self.parent_class}.png")
        if dialog.tabs_list[self.parent_class].is_first == False:
             dialog.tabs_list[self.parent_class].showImage(f"output{self.parent_class}.png")


class PredictDialog(QDialog, QWidget):
    num_grid_rows = 3
    file_name = ""
    num_buttons = 4

    def __init__(self, my_class):
        super().__init__()
        self.my_class:int=my_class
        self.is_first = True
        self.imageLabel = imageQLabel(parent_class=self.my_class)
        self.imageLabel.setPixmap(QtGui.QPixmap())
        self.imageLabel.setStyleSheet("border: 1px solid black;")
        self.imageLabel.setFixedSize(400, 400)

        self._file_menu = None
        self._menu_bar = None
        self._exit_action = None

        self.create_menu()
        self.create_first_button_layout()
        self.create_second_button_layout()

        self.main_layout = QVBoxLayout()
        self.main_layout.setMenuBar(self._menu_bar)
        self.main_layout.addLayout(self.first_button_layout)
        self.main_layout.addWidget(self.imageLabel)
        self.main_layout.addLayout(self.second_button_layout)

        self.setLayout(self.main_layout)

    def predict(self):
        if self.is_first ==False:
            prediction=cm.predict_by_image(f"output{self.my_class}.png")
            keys=list(prediction.keys())
            values=list(prediction.values())
            if float(values[0][:-1]) < 60.0:
                self.predict_label.setText(rf"No class matches your image <br> Try another image")
            else:
                self.predict_label.setText(rf"Your image is:<br> {keys[0]}: {values[0]} <br> {keys[1]}: {values[1]} ")
            gradcam.main()
            # image = cv2.imread(cnfg.dest_dir+"//output0.png")
            # image = cv2.resize(image, (image.width*6, image.height*6), interpolation=cv2.INTER_AREA)
            #
            # cv2.imshow("prediction explaination",image)
            # cv2.waitKey(5000)


    def open_camera(self):
        self.camera = Camera(self.my_class)

    def create_menu(self):
        self._menu_bar = QMenuBar()
        self._file_menu = QMenu("", self)
        self._exit_action = self._file_menu.addAction("&Exit")
        self._menu_bar.addMenu(self._file_menu)
        self._exit_action.triggered.connect(self.accept)

    def create_first_button_layout(self):
        self.first_button_layout = QHBoxLayout()
        button = QPushButton(f"Upload image")
        button.setStyleSheet(" background-color:#68838B	; font-size:18px")
        button.clicked.connect(self.openFileNameDialog)
        self.first_button_layout.addWidget(button)

        button = QPushButton(f"Take a picture")
        button.setStyleSheet("background-color:#68838B	; font-size:18px")
        button.clicked.connect(self.open_camera)
        self.first_button_layout.addWidget(button)

    def create_second_button_layout(self):
        self.second_button_layout = QHBoxLayout()
        button = QPushButton("predict")
        button.setFixedSize(100, 100)
        button.setStyleSheet(" background-color:#68838B	; font-size:20px")
        # button.setText(rf"Predict <br> your <br> image")
        button.clicked.connect(self.predict)
        self.second_button_layout.addWidget(button)

        self.predict_label = QLabel()
        self.predict_label.setText("you will see the predict result here")
        self.predict_label.setFixedSize(290, 100)
        self.predict_label.setStyleSheet("border: 1px solid black; background-color:lightblue; font-size:18px")
        self.predict_label.setAlignment(QtCore.Qt.AlignCenter)
        self.second_button_layout.addWidget(self.predict_label)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getOpenFileName(self, 'Load image', cnfg.open_file_path, 'Motor Files (*.jpg *.png)')[0]
        if (fileName):
            self.showImage(fileName)

    def showImage(self, image_path: str):
        self.is_first=False
        self.predict_label.setText("you will see the predict result here")
        image = Image.open(image_path)
        h = image.height
        w = image.width
        if w > h:
            border = (0, (w-h)//2, 0, (w-h)//2)
            image = ImageOps.expand(image, border=border, fill=RGB(245, 245, 245))
        elif h > w:
            border = ((h-w)//2, 0,  (h-w)//2, 0)
            image = ImageOps.expand(image, border=border, fill=RGB(245, 245, 245))
        image = image.resize((400, 400))
        image.save(f"output{self.my_class}.png")

        pixmap = QtGui.QPixmap(f"output{self.my_class}.png")
        self.imageLabel.setPixmap(pixmap)

class AddImageDialog(QDialog,QWidget ):
    file_name = ""

    def __init__(self, my_class):
        super().__init__()
        self.my_class: int = my_class
        self.is_first = True
        self.imageLabel = imageQLabel(parent_class=self.my_class)
        self.imageLabel.setPixmap(QtGui.QPixmap())
        self.imageLabel.setStyleSheet("border: 1px solid black; ")
        self.imageLabel.setFixedSize(400, 400)

        self._file_menu = None
        self._menu_bar = None
        self._exit_action = None
        self.camera = None

        self.create_menu()
        self.create_first_button_layout()
        self.create_second_button_layout()

        self.main_layout = QVBoxLayout()
        self.main_layout.setMenuBar(self._menu_bar)
        self.main_layout.addLayout(self.first_button_layout)
        self.main_layout.addWidget(self.imageLabel)
        self.main_layout.addLayout(self.second_button_layout)

        self.setLayout(self.main_layout)

    def add_image_to_test(self):
        if self.is_first == False:
            value_label = self.labels_comboBox.currentText()
            labels = cm.load_our_labels()
            timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
            image_name = f"personal-{timestamp}.png"
            cd.insert_personal_image_to_csv(image_name, (list(labels.keys())[list(labels.values()).index(value_label)]))
            self.imageLabel.setPixmap(QtGui.QPixmap())
            self.is_first = True

    def open_camera(self):
        self.camera = Camera(self.my_class)

    def create_menu(self):
        self._menu_bar = QMenuBar()
        self._file_menu = QMenu("", self)
        self._exit_action = self._file_menu.addAction("&Exit")
        self._menu_bar.addMenu(self._file_menu)
        self._exit_action.triggered.connect(self.accept)

    def create_first_button_layout(self):
        self.first_button_layout = QHBoxLayout()
        button = QPushButton(f"Upload image")
        button.setStyleSheet("hieght:30px; background-color:#68838B; font-size:18px")
        button.clicked.connect(self.openFileNameDialog)
        self.first_button_layout.addWidget(button)

        button = QPushButton(f"Take a picture")
        button.setStyleSheet("hieght:30px; background-color:#68838B; font-size:18px")
        button.clicked.connect(self.open_camera)
        self.first_button_layout.addWidget(button)

    def create_second_button_layout(self):
        self.second_button_layout = QGridLayout()
        self.labels_comboBox = QComboBox()
        self.labels_comboBox.setStyleSheet(" font-size:20px")
        self.labels_comboBox.addItems(cm.load_our_labels().values())
        self.second_button_layout.addWidget(self.labels_comboBox,0,0)

        button = QPushButton(f"Add image to test")
        button.setStyleSheet("hieght:30px; background-color:lightblue; font-size:20px")
        button.clicked.connect(self.add_image_to_test)
        self.second_button_layout.addWidget(button,1,0)


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getOpenFileName(self, 'Load image',cnfg.open_file_path, 'Motor Files (*.png *jpg)')[0]
        if fileName:
            self.showImage(fileName)

    def showImage(self, image_path: str):
        self.is_first=False
        image = Image.open(image_path)
        h=image.height
        w=image.width
        if w > h:
            border = (0,(w-h)//2, 0, (w-h)//2)
            image = ImageOps.expand(image, border=border, fill=RGB(245,245,245))
        elif h > w:
            border = ((h-w)//2,0,  (h-w)//2,0)
            image = ImageOps.expand(image, border=border, fill=RGB(245,245,245))
        image=image.resize((400,400))
        image.save(f"output{self.my_class}.png")

        pixmap = QtGui.QPixmap(f"output{self.my_class}.png")
        self.imageLabel.setPixmap(pixmap)



class VisualizationDialog(QDialog):
    def __init__(self, my_class=None):
        QDialog.__init__(self)
        self.my_class=my_class
        self.main_layout = QVBoxLayout()
        self.create_data_visu_layout()
        self.create_model_visu_layout()
        self.main_layout.addWidget(self.data_visu_gbox)
        self.main_layout.addWidget(self.model_visu_gbox)

        self.setLayout(self.main_layout)

    def create_data_visu_layout(self):
        self.data_visu_gbox=QGroupBox("data visualization")
        self.data_visu_gbox.setFixedSize(400,150)
        self.data_visu_gbox.setStyleSheet("font-size:18")

        data_visu = QGridLayout()

        button = QPushButton(f"show splited classes count")
        button.clicked.connect(lambda x: dv.show_splited_classes_count())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD")
        data_visu.addWidget(button,0,0)

        button = QPushButton(f"show classes count")
        button.clicked.connect(lambda x: dv.show_classes_count())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD")
        data_visu.addWidget(button,0,1)

        button = QPushButton(f"show 10 images of class:")
        button.clicked.connect(self.show_10_image_of_class)
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD")
        data_visu.addWidget(button,1,0)

        self.labels_comboBox = QComboBox()
        self.labels_comboBox.addItems(cm.load_our_labels().values())
        self.labels_comboBox.setStyleSheet("hieght:40px ; font-size:16")
        data_visu.addWidget(self.labels_comboBox, 1, 1)

        self.data_visu_gbox.setLayout(data_visu)


    def show_10_image_of_class(self):
        value_label = self.labels_comboBox.currentText()
        labels = cm.load_our_labels()
        dv.show_10_image_of_class(list(labels.keys())[list(labels.values()).index(value_label)])


    def create_model_visu_layout(self):
        self.model_visu_gbox = QGroupBox("model visualization")
        self.model_visu_gbox.setFixedSize(400,150)
        self.model_visu_gbox.setStyleSheet("font-size:18")

        model_visu = QGridLayout()

        button = QPushButton(f"show confusion matrix")
        button.clicked.connect(lambda x: mv.show_confusion_matrix())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD")
        model_visu.addWidget(button,0,0)

        button = QPushButton(f"show predicts samples")
        button.clicked.connect(lambda x: mv.show_samples_after_predict())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD")
        model_visu.addWidget(button,0,1)

        button = QPushButton(f"show wrong predicts ")
        button.clicked.connect(lambda x: mv.show_wrong_predicts())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD ")
        model_visu.addWidget(button,1,0)

        button = QPushButton(f"show plot model history")
        button.clicked.connect(lambda x: mv.plotmodelhistory())
        button.setStyleSheet("hieght:35px; background-color:#7AC5CD ")
        model_visu.addWidget(button,1,1)

        self.model_visu_gbox.setLayout(model_visu)


class TabDialog(QWidget):
    def __init__(self, my_class=None):
        QDialog.__init__(self, my_class)
        layout = QGridLayout()
        self.setLayout(layout)
        self.setWindowTitle("classification project")
        self.tabs_list = [PredictDialog(cnfg.my_class_predict), AddImageDialog(cnfg.my_class_add_image), VisualizationDialog(cnfg.my_class_visu)]
        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.tabs_list[0], "Predict an image ")
        self.tabWidget.addTab(self.tabs_list[1], "Add image to test")
        self.tabWidget.addTab(self.tabs_list[2], "Visualization")
        self.tabWidget.setStyleSheet("font-size:16px")
        layout.addWidget(self.tabWidget, 0, 0)



app = QApplication(sys.argv)
dialog =TabDialog()
dialog.show()
sys.exit(app.exec())
