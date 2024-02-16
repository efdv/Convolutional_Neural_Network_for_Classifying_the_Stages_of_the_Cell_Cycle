import glob
import numpy as np
import cv2

path = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/CellCycle-dataset-efdv/cellcycle_with_wgangp/G1/"
pathSave = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/CellCycle-dataset-efdv/cellcycle_with_wgangp/G1_under_sampling/"

dirfile = glob.glob(path+'/*')
shluffle_indices = np.random.permutation(len(dirfile))
new_indices = shluffle_indices[:8610]


cont = 0
for i in new_indices:
    I = cv2.imread(dirfile[i])
    cv2.imwrite(pathSave+str(cont)+'.png', I)
    print("La imagen %i ha sido reescrita" %(i))
    cont += 1
    

