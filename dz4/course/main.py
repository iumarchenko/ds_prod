import os

os.system('python build_dataset.py')
os.system('python prepare_dataset.py')
os.system('python smote.py')
os.system('python model.py')
os.system('python validation.py')
os.system('python predition.py')