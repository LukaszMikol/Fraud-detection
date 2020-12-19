import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

class Graphs():
    ''' Draws graphs '''
    
    def __init__(self):
        pass
    
    def plot_hist(self, value: object):
        fig = plt.figure(figsize=(8, 8))
        plt.hist(value, bins='auto')  
        plt.title(f"Histogram for {value}")
        plt.show()

    def plot_confusion_matrix(self, y_test: list, y_pred: list, title: str = "Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)

        cm = cm[::-1]
        cm = pd.DataFrame(cm, columns=['Valid', 'Fraud'], index=['Fraud', 'Valid'])

        fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), 
                                              colorscale='ice', showscale=True, reversescale=True)
        fig.update_layout(width=400, height=400, title=title, font_size=16)
        fig.show()
        
    def plot_loss_curves(self, history: object, save_path: str):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title("Loss per epochs")
        plt.ylabel('Loss')
        plt.xlabel('Number epochs')
        plt.legend(['Train data', 'Validation data'], loc="upper right")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        image_name = "Loss_curves.png"
        fig.savefig(os.path.join(save_path, image_name), dpi=fig.dpi)
        plt.show()
        
    def plot_acc_loss():
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(y=metrics.loss, name="loss", mode='lines+markers'), 1, 1)
        fig.add_trace(go.Scatter(y=metrics.val_loss, name="val_loss", mode='lines+markers'), 1, 1)
        fig.add_trace(go.Scatter(y=metrics.accuracy, name="accuracy", mode='lines+markers'), 1, 2)
        fig.add_trace(go.Scatter(y=metrics.val_accuracy, name="val_accuracy", mode='lines+markers'), 1, 2)
        fig.update_xaxes(title_text='epochs')
        fig.update_yaxes(title_text='accuracy')
        fig.update_layout(width=1000, title='Accuracy and Loss')
        fig.show()
