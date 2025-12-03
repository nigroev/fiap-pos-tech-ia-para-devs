from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cursofiaprm369853-package',
    version='1.0.0',
    packages=find_packages(),
    description='Descricao da sua lib cursofiaprm369853',
    author='Paulo Sobral',
    author_email='paulo@paulosobral.com.br',
    url='https://github.com/paulosobral/fiap-pos-tech-ia-para-devs/tree/02-fundamentos-de-ia-e-machine-learning/03-criacao-de-modulos-e-bibliotecas/02-fundamentos-de-ia-e-machine-learning/03-criacao-de-modulos-e-bibliotecas/cursofiaprm369853',  
    license='MIT',  
    long_description=long_description,
    long_description_content_type='text/markdown'
)
