## Veraz
### Un detector de mentiras a control remoto

Truthsayer te permite monitorizar el ritmo cardíaco y posibles 'indicios' de engaño desde cualquier rostro, incluidas videollamadas en directo o grabaciones.

Demostración en vídeo y más información [disponible
aquí](https://youtu.be/5q-BQ2Q_pqI)!

![demostración](demostración.png)

TruthSayer usa [OpenCV] (https://github.com/opencv/opencv-python) y [Face Mesh] de MediaPipe (https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api ) para realizar la detección en tiempo real de puntos de referencia faciales a partir de la entrada de vídeo. También utiliza [FER](https://pypi.org/project/fer/) para la detección del estado de ánimo. A partir de ahí, se calculan las diferencias relativas para determinar cambios significativos en movimientos faciales específicos desde el punto de referencia de una persona, incluido su:

- Ritmo cardiaco
- Frecuencia de parpadeo
- Cambio de mirada
- Mano cubriéndose la cara
- Compresión de labios

Truthsayer puede incluir opcionalmente indicaciones basadas en una segunda transmisión de video para "reflejar" mejor la entrada original.

Hit `Q` on the preview window to exit the resulting display frame, or 
`CTRL+C` at the terminal to close the Python process.


**Truthsayer is built for Python 3 and will not run on 2.x.**


Optional flags:

- `--help` - Display the below options
- `--input` - Choose a camera, video file path, or screen dimensions in the form `x y width height` - defaults to device `0`
- `--landmarks` - Set to any value to draw overlayed facial and hand landmarks
- `--bpm` - Set to any value to include a heart rate tracking chart
- `--flip` - Set to any value to flip along the y-axis for selfie view
- `--landmarks` - Set to any value to draw detected body landmarks from MediaPipe
- `--record` - Set to any value to write the output to a timestamped AVI recording in the current folder
- `--second` - Secondary video input device for mirroring prompts (device number or path)
- `--ttl` - Number of subsequent frames to display a tell; defaults to 30

Example usage:
Pre uso: 
 instala Conda.
 crea un ambiente virutal conda con python 3.9 : PS> conda create --name <namewithuotbrackest> python=3.9 tensorflow-gpu cudatoolkit=9.0 cudnn=8.1.0 conda-forge   
 PS \Truthsayer> conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
 instala tensorflow 2.10 en el ambiente virtual conda : PS E:\Truthsayer> & e:/Truthsayer/.conda/python.exe -m pip install "tensorflow<2.11"
 prueba que Tensorflow se haya instalado : PS E:\Truthsayer> & e:/Truthsayer/.conda/python.exe -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"  
 si tienen problema con tensorflow que no se reoconoce pero ya lo instalaste prueba desinstalando  la version 4.5.5.62 por defecto : PS E:\Truthsayer> & e:/Truthsayer/.conda/python.exe -m pip install -U opencv-python==4.5.5.62      
 y luego instala pandas por separado: PS E:\Truthsayer> & e:/Truthsayer/.conda/python.exe -m pip install pandas
 luego deberia correr: PS E:\Truthsayer> & e:/Truthsayer/.conda/python.exe e:/Truthsayer/intercept.py --input 0 --landmarks 1 --flip 1 --record 1  (imput 0) deberia ser la webcam usb. en windows. 
 
  
 

- `python intercept.py -h` - Show all argument options
- `python intercept.py --input 2 --landmarks 1 --flip 1 --record 1` - Camera device 2; overlay landmarks; flip; generate a recording
- `python intercept.py -i "/Downloads/shakira.mp4" --second 0` - Use video file as input; use camera device 0 as secondary input for mirroring feedback
