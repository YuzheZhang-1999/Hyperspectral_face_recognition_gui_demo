from system_operation import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()

    palette = QtGui.QPalette()  # set window background color
    palette.setColor(QtGui.QPalette.Background, QtCore.Qt.white)
    myWin.setPalette(palette)

    myWin.show()
    sys.exit(app.exec_())
