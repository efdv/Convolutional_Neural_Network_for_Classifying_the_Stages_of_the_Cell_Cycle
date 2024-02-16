from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay 

def normResults(y_pred):
    yp = []
    for row in y_pred:
        valmax = max(row)
        for i in range(len(row)):
            if row[i] == valmax:
                row[i] = 1
            else:
                row[i] = 0
        
        yp.append(row)
    
    return yp
 

# data = np.load("../../../datasets/cellcycle_dataset_ch4/datasetRana1/cellCycle.npy")
# labels = np.load("../../../datasets/cellcycle_dataset_ch4/datasetRana1/labels.npy")

data = np.load("../../../datasets/CellCycle-dataset-efdv/cellCycle.npy")
labels = np.load("../../../datasets/CellCycle-dataset-efdv/labels.npy")

# data = np.load("../../../datasets/CellCycle-dataset-efdv/cellcycle_with_wgangp/cellCycle.npy")
# labels = np.load("../../../datasets/CellCycle-dataset-efdv/cellcycle_with_wgangp/labels.npy")

data = data/255.0

skfolds = StratifiedKFold(n_splits=5)

num_fold = 0
for train_index, test_index in skfolds.split(data, labels):
    
    X_train_folds = data[train_index]
    y_train_folds = labels[train_index]
    X_test_folds  = data[test_index]
    y_test_folds  = labels[test_index]
    
    y_train_folds = to_categorical(y_train_folds)
    y_test_folds = to_categorical(y_test_folds)
                                       
    #train
    #model = load_model('E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/projects/cell_cycle/modelos/model_CNN_with_wgandiv_3_2.h5')
    model = load_model('E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/projects/cell_cycle/modelos/model_CNN_3_2.h5')
    # model = load_model('E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/projects/cell_cycle/modelos/model_CNN_with_wgangp_3_2.h5')

    
    y_pred = model.predict(X_test_folds)
    
    ypred_cm = []
    y_test_folds_cm = []
    for i in range(len(y_pred)):
        ypred_cm.append(np.argmax(y_pred[i]))
        y_test_folds_cm.append(np.argmax(y_test_folds[i]))
        
    yp = [] 
    yp = normResults(y_pred)
    
    results = []
    n_correct = sum(yp == y_test_folds)
    results.append(n_correct/len(y_pred))
        
    # rootcm = "../graphs/cm/ch3-wgangp/cm_"
    # rootcmN = "../graphs/cm/ch3-wgangp/cmN_"
    #rootcm = "../graphs/cm/original/cm_"
    #rootcmN = "../graphs/cm/original/cmN_"
    # rootcm = "../graphs/cm/ch4-wgandiv-mixup/cm_"
    # rootcmN = "../graphs/cm/ch4-wgandiv-mixup/cmN_"
    #nameimg = "Confusion matrix" 
    #class_names = ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']
    #cm_display = ConfusionMatrixDisplay.from_predictions(y_test_folds_cm,ypred_cm, cmap=plt.cm.PuBu)
    #cm_display.ax_.set_xticklabels(class_names, fontsize=12)
    #cm_display.ax_.set_yticklabels(class_names, fontsize=12)
    
    # cm_display.figure_.savefig(rootcm + str(num_fold) + '_' + str(2) + '.png', dpi=300)
    # cm_display.figure_.savefig(rootcm + str(num_fold) + '_' + str(2) + '.eps', dpi=300)


    cm1_display = ConfusionMatrixDisplay.from_predictions(y_test_folds_cm,ypred_cm, cmap=plt.cm.PuBu, normalize="true", values_format=".3f")
    cm1_display.ax_.set_xticklabels(class_names)
    cm1_display.ax_.set_yticklabels(class_names)    
    
    
    cm1_display.figure_.savefig(rootcmN + str(num_fold) + '_' + str(2) + '.png', dpi=300)
    cm1_display.figure_.savefig(rootcmN + str(num_fold) + '_' + str(2) + '.eps', dpi=300)
    
    print(num_fold)
    num_fold += 1
