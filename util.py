import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QSystemTrayIcon, QMenu, QAction
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QStyle
import cv2


class CameraWorker(QObject):
    finished = pyqtSignal()
    frame_ready = pyqtSignal(object)

    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_ready.emit(frame)
        cap.release()
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... your main window UI setup ...

        # --- System Tray Icon Setup ---
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        show_action = QAction("Show", self)
        quit_action = QAction("Exit", self)
        hide_action = QAction("Hide", self)

        show_action.triggered.connect(self.show)
        hide_action.triggered.connect(self.hide)
        quit_action.triggered.connect(QApplication.instance().quit)

        tray_menu = QMenu()
        tray_menu.addAction(show_action)
        tray_menu.addAction(hide_action)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # --- Camera Thread Setup ---
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self.process_frame)
        self.camera_thread.start()

    def process_frame(self, frame):
        # This is where you would handle the captured frame
        # For example, display it in the UI if the window is visible
        # or perform some background analysis.
        # This method will be called continuously from the background thread.
        pass

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Application Minimized",
            "The application is still running in the tray.",
            QSystemTrayIcon.Information,
            2000
        )
