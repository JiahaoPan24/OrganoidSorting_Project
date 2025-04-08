import typing
from pathlib import Path
from typing import List
from PySide6.QtWidgets import QMainWindow, QWidget, QListWidget, QPushButton, QVBoxLayout, \
    QHBoxLayout, QFileDialog, QLabel, QFormLayout, QSpinBox, QLineEdit, QCheckBox, \
    QDoubleSpinBox, QDialog, QComboBox, QMessageBox, QSlider, QDialogButtonBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer
from PIL import ImageQt, Image
from UI.ProcessingWorker import ProcessingWorker
from UI.WindowProcessingWorker import WindowProcessingWorker
import os



class MainWindow(QMainWindow):
    Instance: 'MainWindow' = None

    def __init__(self):
        super().__init__()
        MainWindow.Instance = self
        self.fileDialog = FileListWidget()
        self.settingsWidget = SettingsWidget()
        self.goButton = QPushButton("Process Images")
        self.goButton.clicked.connect(self.Process)
        self.windowButton = QPushButton("Run Yolo on Window")
        self.windowButton.clicked.connect(self.WindowProcess)
        self.processingWorker = ProcessingWorker()
        self.windowProcessingWorker = WindowProcessingWorker()

        mainWidget = QWidget()
        mainLayout = QVBoxLayout()
        threeLayout = QHBoxLayout()
        mainLayout.addLayout(threeLayout)
        mainLayout.addWidget(self.goButton)
        mainLayout.addWidget(self.windowButton)
        mainWidget.setLayout(mainLayout)
        threeLayout.addWidget(self.fileDialog)
        threeLayout.addWidget(self.settingsWidget)
        self.setWindowTitle("Organoid Detection")
        self.setCentralWidget(mainWidget)

    def Process(self):
        self.ExecuteOnImagesAsync(self.fileDialog.paths, True,
                                  lambda x: QMessageBox.information(self, "Job Completed",
                                                                    "Images have been processed."))
        
    def WindowProcess(self):
        dialog = WindowNameDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            window_name = dialog.get_window_name()
            self.WindowExecute(window_name)
        else:
            pass
        

    def ExecuteOnImagesAsync(self, paths: List[Path], output: bool,
                             finishedDelegate: typing.Callable):
        settings = [self.settingsWidget.runClassificationWidget.isChecked(),
                    Path(self.settingsWidget.modelPathWidget.text()),
                    paths,
                    Path(self.settingsWidget.outputPathWidget.text()) if output else None,
                    Path(self.settingsWidget.classModelPathWidget.text()),
                    self.settingsWidget.thresholdWidget.value(),
                    self.settingsWidget.batchSizeWidget.value(),
                    self.settingsWidget.edgeSigmaWidget.value(),
                    self.settingsWidget.edgeMinWidget.value(),
                    self.settingsWidget.edgeMaxWidget.value(),
                    self.settingsWidget.minimumAreaWidget.value(),
                    self.settingsWidget.fillHolesWidget,
                    self.settingsWidget.removeBorderWidget,
                    self.settingsWidget.beliefWidget.isChecked(),
                    self.settingsWidget.binaryWidget.isChecked(),
                    self.settingsWidget.separateContours.isChecked(),
                    self.settingsWidget.edgeWidget,
                    self.settingsWidget.colorizeWidget,
                    self.settingsWidget.labeledWidget.isChecked(),
                    self.settingsWidget.trackWidget.isChecked(),
                    self.settingsWidget.overlayWidget.isChecked(),
                    self.settingsWidget.gifWidget,
                    self.settingsWidget.batchWidget.isChecked(),
                    self.settingsWidget.regionProperties]
        ProcessingDialog(self, settings, finishedDelegate)

    def WindowExecute(self, window_name):
        settings = [window_name,
                    Path(self.settingsWidget.yoloModelPathWidget.text()),
                    []
                    ]
        WindowProcessingDialog(self, settings)


class ProcessingDialog(QDialog):
    def __init__(self, parent, settings, finishedDelegate):
        super().__init__(parent)
        self.timer = QTimer(self)
        MainWindow.Instance.processingWorker.Process(settings)
        self.field = QLabel()
        self.timer.timeout.connect(self.Update)
        self.timer.start(100)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.finishedDelegate = finishedDelegate
        layout.addWidget(self.field)
        self.setModal(True)
        self.show()

    def reject(self) -> None:
        res = QMessageBox.question(self, "Cancel confirmation",
                                   "Are you sure that you want to stop the job?")
        if res == QMessageBox.Yes:
            super().reject()
            MainWindow.Instance.processingWorker.ForceStop()
            MainWindow.Instance.processingWorker = ProcessingWorker()

    @staticmethod
    def TranslateString(input_string):

        # Initial state
        # String is stored as a list because
        # python forbids the modification of
        # a string
        displayed_string = ""

        # Loop on our input (transitions sequence)
        for character in input_string:
            # Backward transition
            if character == "\b":
                displayed_string = displayed_string[:-1]
            else:
                displayed_string = displayed_string + character

        # We transform our "list" string back to a real string
        return displayed_string

    def Update(self):
        self.field.setText(
            self.TranslateString(MainWindow.Instance.processingWorker.GetOutputText()))
        if MainWindow.Instance.processingWorker.HasResults():
            self.timer.stop()
            self.done(0)
            self.finishedDelegate(MainWindow.Instance.processingWorker.Results())


class WindowProcessingDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.timer = QTimer(self)
        MainWindow.Instance.windowProcessingWorker.Process(settings)
        self.field = QLabel()
        self.timer.timeout.connect(self.Update)
        self.timer.start(100)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.field)
        self.setModal(True)
        self.show()

    def reject(self) -> None:
        res = QMessageBox.question(self, "Cancel confirmation",
                                   "Are you sure that you want to stop the job?")
        if res == QMessageBox.Yes:
            super().reject()
            MainWindow.Instance.windowProcessingWorker.ForceStop()
            MainWindow.Instance.windowProcessingWorker = WindowProcessingWorker()

    @staticmethod
    def TranslateString(input_string):

        # Initial state
        # String is stored as a list because
        # python forbids the modification of
        # a string
        displayed_string = ""

        # Loop on our input (transitions sequence)
        for character in input_string:
            # Backward transition
            if character == "\b":
                displayed_string = displayed_string[:-1]
            else:
                displayed_string = displayed_string + character

        # We transform our "list" string back to a real string
        return displayed_string

    def Update(self):
        self.field.setText(
            self.TranslateString(MainWindow.Instance.windowProcessingWorker.GetOutputText()))
        if MainWindow.Instance.windowProcessingWorker.HasResults():
            self.timer.stop()
            self.done(0)
            self.finishedDelegate(MainWindow.Instance.windowProcessingWorker.Results())


class WindowNameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Window Name")

        # Create a layout and add a QLineEdit for the window name
        layout = QVBoxLayout()
        self.windowNameEdit = QLineEdit()
        layout.addWidget(QLabel("Window Name:"))
        layout.addWidget(self.windowNameEdit)

        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_window_name(self):
        return self.windowNameEdit.text()



class SettingsWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Model Settings</b>"), alignment=Qt.AlignTop)

        mainFormLayout = QFormLayout()
        outputFormLayout = QVBoxLayout()
        trackingFormLayout = QVBoxLayout()
        layout.addLayout(mainFormLayout)
        layout.addSpacing(10)
        layout.addWidget(QLabel("<b>Output Option</b>"))
        layout.addLayout(outputFormLayout)
        layout.addSpacing(10)
        #layout.addWidget(QLabel("<b>Tracking Settings</b>"))
        #layout.addLayout(trackingFormLayout)
        self.modelPathWidget = QLineEdit(str(Path(os.getcwd()) / "OptimizedModel"))
        self.classModelPathWidget = QLineEdit(str(Path(os.getcwd()) / "test_model"))
        self.yoloModelPathWidget = QLineEdit(str(Path(os.getcwd()) / "yolo.pt"))
        #self.browseModelWidget = QPushButton("Browse...")
        self.outputPathWidget = QLineEdit(str(Path(os.getcwd())))
        self.browseOutputWidget = QPushButton("Browse...")
        self.browseOutputWidget.clicked.connect(self.BrowseOutput)
        self.thresholdWidget = DoubleSpinBoxWidget(0.5, 0, 1, 0.1)
        self.batchSizeWidget = SpinBoxWidget(16, 1, 128, 1)
        self.edgeSigmaWidget = DoubleSpinBoxWidget(2, 0, 256, 1)
        self.edgeMinWidget = DoubleSpinBoxWidget(0.005, 0, 1, 0.1)
        self.edgeMaxWidget = DoubleSpinBoxWidget(0.05, 0, 1, 0.1)
        self.minimumAreaWidget = SpinBoxWidget(100, 0, 1e9, 100)
        self.fillHolesWidget = True
        self.removeBorderWidget = False
        self.separateContours = CheckBoxWidget("", True)
        self.beliefWidget = CheckBoxWidget("Detection Image", False)
        self.binaryWidget = CheckBoxWidget("Size Distribution", False)
        self.edgeWidget = False
        self.colorizeWidget = False
        self.labeledWidget = CheckBoxWidget("Labeled Image", True)
        self.overlayWidget = CheckBoxWidget("Overlay Image", False)
        self.runClassificationWidget = CheckBoxWidget("Classified Image", False)
        #self.fillHolesWidget = CheckBoxWidget("Fill Holes", True)
        #self.removeBorderWidget = CheckBoxWidget("Remove Border", False)
        #self.edgeWidget = CheckBoxWidget("Edge Image", False)
        #self.colorizeWidget = CheckBoxWidget("Colorize", False)
        #self.gifWidget = CheckBoxWidget("GIF Output", False)
        #self.regionProperties = CheckBoxWidget("Region Properties", False)

        self.gifWidget = False
        self.regionProperties = False

        self.trackWidget = CheckBoxWidget("Track as timelapse", False)
        self.batchWidget = CheckBoxWidget("Batch Tracking", False)

        outputLayout = QHBoxLayout()
        outputLayout.addWidget(self.outputPathWidget)
        outputLayout.addWidget(self.browseOutputWidget)
        #mainFormLayout.addRow("Model Path", self.modelPathWidget)
        mainFormLayout.addRow("Threshold (0-1)", self.thresholdWidget)
        mainFormLayout.addRow("Batch Size", self.batchSizeWidget)
        mainFormLayout.addRow("Minimum Area (px)", self.minimumAreaWidget)
        mainFormLayout.addRow("Edge Sigma (px)", self.edgeSigmaWidget)
        mainFormLayout.addRow("Edge Minimum (0-1)", self.edgeMinWidget)
        mainFormLayout.addRow("Edge Maximum (0-1)", self.edgeMaxWidget)
        #mainFormLayout.addRow("Fill Holes", self.fillHolesWidget)
        #mainFormLayout.addRow("Remove Border", self.removeBorderWidget)
        mainFormLayout.addRow("Separate Contours", self.separateContours)
        #mainFormLayout.addRow("Output Path", outputLayout)

        [outputFormLayout.addWidget(x) for x in [#self.beliefWidget,
                                                 self.binaryWidget,
                                                 #self.edgeWidget,
                                                 #self.colorizeWidget,
                                                 self.overlayWidget,
                                                 #self.gifWidget,
                                                 self.labeledWidget,
                                                 self.runClassificationWidget,
                                                 #self.regionProperties]
        ]]
        # ------------------------------
        self.advancedToggleBtn = QPushButton("Show Advanced Settings ▼")
        self.advancedToggleBtn.setCheckable(True)
        self.advancedToggleBtn.setChecked(False)
        self.advancedToggleBtn.clicked.connect(self.toggleAdvancedSettings)

        self.advancedWidget = QWidget()
        self.advancedWidget.setVisible(False)
        self.advancedLayout = QFormLayout()
        self.advancedWidget.setLayout(self.advancedLayout)
                              

        self.advancedLayout.addRow("Model Path", self.modelPathWidget)
        self.advancedLayout.addRow("Class Model Path", self.classModelPathWidget)
        self.advancedLayout.addRow("YOLO Model Path", self.yoloModelPathWidget)
        
        # outputPathLayout = QHBoxLayout()
        # outputPathLayout.addWidget(self.outputPathWidget)
        # outputPathLayout.addWidget(self.browseOutputWidget)
        self.advancedLayout.addRow("Output Path", outputLayout)


        # [self.advancedLayout.addRow(widget) for widget in [
        #     self.beliefWidget,
        #     self.edgeWidget,
        #     self.colorizeWidget,
        #     self.gifWidget,
        #     self.regionProperties,
        #     self.fillHolesWidget,
        #     self.removeBorderWidget,
        #     self.trackWidget,
        #     self.batchWidget
        # ]]

        layout.addWidget(self.advancedToggleBtn)
        layout.addWidget(self.advancedWidget)
        # ------------------------------

        #[trackingFormLayout.addWidget(x) for x in [self.trackWidget,
        #                                           self.batchWidget]]
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.Verify)
        self.timer.start(30)

    
    def toggleAdvancedSettings(self):
        visible = self.advancedToggleBtn.isChecked()
        self.advancedWidget.setVisible(visible)
        if visible:
            self.advancedToggleBtn.setText("Hide Advanced Settings ▲")
        else:
            self.advancedToggleBtn.setText("Show Advanced Settings ▼")

        self.adjustSize()
        self.updateGeometry()

        hint_height = self.sizeHint().height()
        self.setMinimumHeight(hint_height)
        self.setMaximumHeight(hint_height)

    def BrowseOutput(self):
        directory = QFileDialog.getExistingDirectory(self)
        newPath = Path(directory)
        self.outputPathWidget.setText(str(newPath.absolute()))

    def Verify(self):
        [x.setEnabled(self.separateContours.isChecked()) for x in [#self.edgeWidget,
                                                                   self.edgeSigmaWidget,
                                                                   self.edgeMinWidget,
                                                                   self.edgeMaxWidget]]

        self.batchWidget.setEnabled(self.trackWidget.isChecked())

    def create_int_slider(self, initial, minimum, maximum, label_text):
        slider_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(initial)
        slider_label = QLabel(f"{label_text} {initial}")
        slider.valueChanged.connect(lambda value, lbl=slider_label: lbl.setText(f"{label_text} {value}"))
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        return slider, slider_layout  # 返回滑动条及其布局

    def create_double_slider(self, initial, minimum, maximum, step, label_text):
        slider_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        scale = 1 / step
        slider.setMinimum(int(minimum * scale))
        slider.setMaximum(int(maximum * scale))
        slider.setValue(int(initial * scale))
        slider_label = QLabel(f"{label_text} {initial:.2f}")
        slider.valueChanged.connect(lambda value, lbl=slider_label, scale=scale: lbl.setText(f"{label_text} {value / scale:.2f}"))
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        return slider, slider_layout  # 返回滑动条及其布局




class CheckBoxWidget(QCheckBox):
    def __init__(self, text, initial):
        super().__init__(text)
        self.setChecked(initial)


class SpinBoxWidget(QSpinBox):
    def __init__(self, initial, minimum, maximum, increment):
        super().__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(initial)
        self.setSingleStep(increment)


class DoubleSpinBoxWidget(QDoubleSpinBox):
    def __init__(self, initial, minimum, maximum, increment):
        super().__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(initial)
        self.setSingleStep(increment)


class PreviewDialog(QDialog):
    def __init__(self, parent, results):
        super().__init__(parent)
        self.setWindowTitle("Finished!")
        self.results = results
        self.resultsDropdown = QComboBox()
        self.resultsDropdown.addItems(list(self.results))
        self.resultsDropdown.setCurrentIndex(len(self.results) - 1)
        self.resultsDropdown.currentIndexChanged.connect(self.ResultChanged)
        self.frameDropdown = QComboBox()
        self.frameDropdown.currentIndexChanged.connect(self.FrameChanged)

        self.imageLabel = QLabel()
        form = QFormLayout()
        form.addRow("Result", self.resultsDropdown)
        form.addRow("Frame", self.frameDropdown)
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(form)
        mainLayout.addWidget(self.imageLabel)
        self.setLayout(mainLayout)
        self.setModal(False)
        self.ResultChanged()
        self.show()

    def ResultChanged(self):
        currentImage = self.results[self.resultsDropdown.currentText()]
        self.frameDropdown.clear()
        self.frameDropdown.addItems([str(i) for i in range(currentImage.shape[0])])
        self.FrameChanged()

    def FrameChanged(self):
        currentImage = self.results[self.resultsDropdown.currentText()][
            self.frameDropdown.currentIndex()]
        image = QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(currentImage)))
        self.imageLabel.setPixmap(image)


class FileListWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.fileView = QListWidget()
        self.addFilesButton = QPushButton("Add Files...")
        self.addFilesButton.clicked.connect(self.AddFiles)
        self.addFolderButton = QPushButton("Add Folder...")
        self.addFolderButton.clicked.connect(self.AddFolder)
        self.removeButton = QPushButton("Remove")
        self.removeButton.clicked.connect(self.Remove)
        self.previewButton = QPushButton("Preview\nSelected")
        self.previewButton.clicked.connect(self.Preview)
        self.fileView.setSelectionMode(QListWidget.ExtendedSelection)
        self.fileView.itemSelectionChanged.connect(self.ListSelectionChanged)
        layout0 = QVBoxLayout()
        layout0.addWidget(QLabel("<b>Files</b>"), alignment=Qt.AlignTop)
        layout1 = QHBoxLayout()
        layout0.addLayout(layout1)
        layout1.addWidget(self.fileView)
        layout2 = QVBoxLayout()
        layout1.addLayout(layout2)
        layout2.addWidget(self.addFilesButton)
        layout2.addWidget(self.addFolderButton)
        layout2.addWidget(self.removeButton)
        layout2.addWidget(self.previewButton, alignment=Qt.AlignBottom)
        self.paths = []
        self.setLayout(layout0)
        self.ListSelectionChanged()

    def Preview(self):
        indices = [self.fileView.row(i) for i in self.fileView.selectedItems()]
        pathsToProcess = [self.paths[i] for i in indices]
        MainWindow.Instance.ExecuteOnImagesAsync(pathsToProcess, False, 
                                                 lambda results: PreviewDialog(self, results))

    def ListSelectionChanged(self):
        self.removeButton.setEnabled(len(self.fileView.selectedItems()) > 0)
        self.previewButton.setEnabled(len(self.fileView.selectedItems()) > 0)

    def AddFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self)
        newPaths = [Path(p) for p in files]
        self.paths += newPaths
        self.fileView.addItems([str(p.absolute()) for p in newPaths])

    def AddFolder(self):
        directory = QFileDialog.getExistingDirectory(self)
        newPath = Path(directory)
        self.paths.append(newPath)
        self.fileView.addItem(str(newPath.absolute()))

    def Remove(self):
        indices = [self.fileView.row(i) for i in self.fileView.selectedItems()]
        [self.fileView.takeItem(i) for i in indices]
        self.paths = [p for i, p in enumerate(self.paths) if i not in indices]