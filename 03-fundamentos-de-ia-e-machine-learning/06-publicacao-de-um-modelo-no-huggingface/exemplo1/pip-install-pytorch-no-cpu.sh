./venv/bin/python -m pip cache purge
./venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
./venv/bin/pip install -r requirements.txt --no-cache-dir