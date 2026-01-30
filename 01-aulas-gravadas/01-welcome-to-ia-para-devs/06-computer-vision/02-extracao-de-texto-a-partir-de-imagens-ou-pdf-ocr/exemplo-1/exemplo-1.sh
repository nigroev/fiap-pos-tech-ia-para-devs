# Ubuntu
# sudo apt-get -y install tesseract-ocr
# sudo apt-get -y install libtesseract-dev
# sudo apt-get -y install tesseract-ocr-por

# Fedora
sudo dnf install -y tesseract
sudo dnf install -y tesseract-devel leptonica-devel
sudo dnf install -y tesseract-langpack-por

GDK_BACKEND=x11 ./venv/bin/python exemplo-1.py