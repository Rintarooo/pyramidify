#!/bin/bash

# pyenvのバージョン
# pyenv --version
pyenv -v
# pyenvでインストールしたpythonのバージョン
pyenv versions
# pyenv install --list | grep -i "3.9."
# pyenv install 3.9.1
# pyenv local 3.9.1
pyenv install --list | grep -i "3.10."
pyenv install 3.10.1
pyenv local 3.10.1
pyenv versions

poetry init
poetry shell
# シェルに入った状態で
python -V

poetry add numpy matplotlib plotly 
poetry add scipy

# poetry run python sample.py