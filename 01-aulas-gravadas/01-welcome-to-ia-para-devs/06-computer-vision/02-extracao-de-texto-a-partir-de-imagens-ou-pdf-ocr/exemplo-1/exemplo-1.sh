# Ubuntu
# sudo apt-get -y install tesseract-ocr
# sudo apt-get -y install libtesseract-dev
# sudo apt-get -y install tesseract-ocr-por

# Fedora
# sudo dnf install -y tesseract
# sudo dnf install -y tesseract-devel leptonica-devel
# sudo dnf install -y tesseract-langpack-por
# ln -snf /usr/share/fonts ./venv/lib/python3.14/site-packages/cv2/qt/fonts

# Ensure Qt can find system fonts (avoid QFontDatabase warnings)
GDK_BACKEND=x11 ./venv/bin/python exemplo-1.py