# Create a virtual environment
python3 -m venv fruit_classifier_venv

# Activate the virtual environment
source ./fruit_classifier_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf fruit_classifier_venv