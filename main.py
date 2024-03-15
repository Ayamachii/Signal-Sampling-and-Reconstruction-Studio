import sys
from PyQt6.QtWidgets import QApplication

from MainWindow import MyMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.setWindowTitle("Sampling Studio")
    mainWindow.show()
    sys.exit(app.exec())
