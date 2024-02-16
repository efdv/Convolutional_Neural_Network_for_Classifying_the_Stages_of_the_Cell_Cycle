
import glob
import cv2
import os

path = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/CellCycle/"
pathSave = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/cellcycle_dataset_ch4/"
nameFolders = ['G1', 'S', 'G2', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']
wordkey = 'Ch4'

for phase in nameFolders:
    os.makedirs(pathSave+phase, exist_ok=True)
    print(phase)
    dirfile =  glob.glob(path+phase+'/*')
    cont = 0
    p = 0
    for ipath in dirfile:
        porcentaje = (p*100)/len(dirfile)
        if wordkey in ipath:
            cont += 1 
            I = cv2.imread(ipath)
            resized_image = cv2.resize(I, (64,64))
            cv2.imwrite(pathSave+phase+'/'+str(cont)+'.png', resized_image)
            print("%.2d completado de %i"%(porcentaje, len(dirfile)))
            print("La imagen del directorio %s ha sido gardada en el directorio %s" %(path+phase, pathSave+phase))
        p + 1
    