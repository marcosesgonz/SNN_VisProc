# SNN_VisProc

Se presenta el código para el uso de los conjuntos de datos *SL-Animals, DVSGesture, Action Recognition (AR), MAD y Daily Action* en la clasificación de acciones. 
- Se utiliza el emulador e2vid de *"Rebecq, H., Ranftl, R., Koltun, V., and Scaramuzza, D. (2019). High speed and
high dynamic range video with an event camera. IEEE Trans. Pattern Anal. Mach.
Intell. (T-PAMI)."* para poder formar datos de vídeo en estos conjuntos de datos neuromórficos.
- Las redes neuronales implementadas son la presente en *"Leaky integrate-and-fire spiking neuron with learnable membrane
time parameter"* con el nombre **DVSGnet**,y la red **SEW-Resnet18** y  **7Bnet** presentes en *"Fang, W., Yu, Z., Chen, Y., Huang, T., Masquelier, T., and Tian, Y. (2022). Deep
residual learning in spiking neural networks."*.
- Los métodos de integración presentados para los eventos son la *integración por número, por tiempo y el decaimiento exponencial*.

En **src/Laboratory.py** se presentan las principales funciones utilizadas para la ejecución de experimentos de una forma simple con estos conjuntos de datos.

En **data/.**.  se encuentra las carpetas vacías que se deben rellenar con la descarga de los datos de internet para los distintos conjuntos de datos.

