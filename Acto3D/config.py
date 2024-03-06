# /Acto3D/config.py
class Config:
    def __init__(self):
        self.host = 'localhost'
        self.port = 41233
        self.path_to_Acto3D = '/Applications/Acto3D.app'
        
    def print_params(self):
        print(f'host:{self.host}, port:{self.port}, path:{self.path_to_Acto3D}')


config = Config()