import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.

def plots(trainData,validData,ylabel,title,trainLabel,validLabel):
	# epochs = list(range(len(trainData)))
	plt.plot(list(range(len(trainData))),trainData,label=trainLabel)
	plt.plot(list(range(len(validData))),validData,label=validLabel)
	plt.xlabel('epochs')
	plt.ylabel(ylabel)
	plt.legend()
	plt.title(title)
	plt.savefig(ylabel+'.jpg')
	plt.close()

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plots(train_losses,valid_losses,'Loss','Loss Curve','Training Loss','Validation Loss')
	plots(train_accuracies,valid_accuracies,'Accuracy','Accuracy Curve','Training Accuracy','Validation Accuracy')
	pass


	# Confusion matrix plot from scikit-learn example
	# taken from
	# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix1(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.jpg')

def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# print(results)
	y_true,y_pred,_ = zip(*results)
	cnf_matrix = confusion_matrix(y_true, y_pred)
	plot_confusion_matrix1(cnf_matrix,class_names)
