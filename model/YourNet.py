# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from model.CNNBaseModel import CNNBaseModel
from layers.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourNet.  Le réseau est constitué de

    1) quelques blocs d'opérations de base du type «conv-batch-norm-relu»
    2) 1 (ou plus) bloc dense inspiré du modèle «denseNet)
    3) 1 (ou plus) bloc résiduel inspiré de «resNet»
    4) 1 (ou plus) bloc de couches «bottleneck» avec ou sans connexion résiduelle, c’est au choix
    5) 1 (ou plus) couches pleinement connectées 
    
    NOTE : le code des blocks résiduel, dense et bottleneck doivent être mis dans le fichier CNNBlocks.py
    Aussi, vous pouvez vous inspirer du code de CNNVanilla.py pour celui de *YourNet*

'''


class YourNet(CNNBaseModel):

    def __init__(self):
        super(YourNet, self).__init__()


'''
FIN DE VOTRE CODE
'''
