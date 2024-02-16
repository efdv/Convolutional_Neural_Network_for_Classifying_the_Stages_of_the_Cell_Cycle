import tensorflow as tf

# Verificar las GPUs disponibles
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU(s) disponibles:", len(gpus))
else:
    print("No se encontraron GPUs disponibles.")