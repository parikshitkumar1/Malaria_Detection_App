import streamlit as st
from PIL import Image
from model import predict

import torch.nn as nn

###################################################################
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class MalariaModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(), 
            nn.Linear(64*28*28, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.Softmax())
        
    def forward(self, xb):
        return self.network(xb)

#######################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Malaria Detector")

st.write("""Uses Cell Images to detect if a person has malaria or not:\n
            0. Parasitized
            1. Uninfected """)
st.write("""Dataset:\n
            https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria""")
st.write("")

file_up = st.file_uploader("Upload a cell Image", type=["png","jpg","jpeg"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption=' Please wait for a few seconds...', use_column_width=True)
    st.write("")
    st.write("Prediction:   ")
    labels = predict(file_up)

    # print out the top 3 prediction labels with scores
    for i in labels:
        st.write(" Probability of ",i[0], i[1])
