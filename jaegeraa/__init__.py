import os 

if not os.environ.get('TF_CPP_MIN_LOG_LEVEL',None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

__version__ = '1.1.23'