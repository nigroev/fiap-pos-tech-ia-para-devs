#!/bin/bash

# Verifica se um nome foi fornecido
if [ -z "$1" ]; then
    echo "Uso: $0 nome_da_estrutura"
    exit 1
fi

# Nome da estrutura
STRUCTURE_NAME=$1

# Cria a estrutura de diretórios e arquivos
mkdir -p $STRUCTURE_NAME/$STRUCTURE_NAME
touch $STRUCTURE_NAME/$STRUCTURE_NAME/__init__.py

# Adiciona o código da biblioteca no core.py
cat <<EOL > $STRUCTURE_NAME/$STRUCTURE_NAME/core.py
# $STRUCTURE_NAME/core.py

def hello_world():
    return "Hello, world!"
EOL

mkdir -p $STRUCTURE_NAME/tests

# Adiciona um teste básico no test_core.py
cat <<EOL > $STRUCTURE_NAME/tests/test_core.py
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from $STRUCTURE_NAME.core import hello_world

def test_hello_world():
    assert hello_world() == "Hello, world!"
EOL

# Cria o setup.py com a configuração do pacote
cat <<EOL > $STRUCTURE_NAME/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='${STRUCTURE_NAME}-package',
    version='1.0.0',
    packages=find_packages(),
    description='Descricao da sua lib ${STRUCTURE_NAME}',
    author='seu nome',
    author_email='seu.email@example.com',
    url='https://github.com/tadrianonet/${STRUCTURE_NAME}',  
    license='MIT',  
    long_description=long_description,
    long_description_content_type='text/markdown'
)
EOL

# Cria o README.md com instruções básicas
cat <<EOL > $STRUCTURE_NAME/README.md
# ${STRUCTURE_NAME^}

A simple example library.

## Installation

\`\`\`sh
pip install ${STRUCTURE_NAME}
\`\`\`

## Usage

\`\`\`python
from ${STRUCTURE_NAME} import hello_world

print(hello_world())
\`\`\`
EOL

echo "Estrutura $STRUCTURE_NAME criada com sucesso!"

# Depois:
# 1-) Prencher o setup.py e conigura o token do pypi.org
# 2-) Executar: python setup.py sdist bdist_wheel
# 3-) Executar: twine upload dist/*