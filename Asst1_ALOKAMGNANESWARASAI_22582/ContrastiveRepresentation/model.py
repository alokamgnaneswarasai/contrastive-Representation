import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # TODO: fill in this class with the required architecture and
    # TODO: associated forward method
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        
        self.conv = nn.Sequential( 
            nn.Conv2d(3,64,kernel_size=3, padding="same"),# padding="same" equivalent to padding=2 when kernel_size=5
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
                                  
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), # padding="same" equivalent to padding=2 when kernel_size=5
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
                                                             
            nn.Conv2d(64, 128, kernel_size=3, padding="same"), # padding="same" equivalent to padding=2 when kernel_size=5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding="same"), # padding="same" equivalent to padding=2 when kernel_size=5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            
           
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
           )
        
        self.fc = nn.Sequential(
             nn.Linear(512 , 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, z_dim))
        
       
        
        
    def forward(self, x):
        
        
        x = self.conv(x)
      
        x = torch.flatten(x, 1)  
        x = self.fc(x)
        return x
    

class Classifier(nn.Module):
    # TODO: fill in this class with the required architecture and
    # TODO: associated forward method
    def __init__(self, z_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 10),
        )
        
    def forward(self, x):
       
        x = self.fc(x)
    
        return x
    