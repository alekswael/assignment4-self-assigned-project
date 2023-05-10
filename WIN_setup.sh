# Create a virtual environment
python -m venv fruit_classifier_venv

# Activate the virtual environment
source ./fruit_classifier_venv/Scripts/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf fruit_classifier_venv