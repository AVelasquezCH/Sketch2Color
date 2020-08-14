<h1 align="center">Desarrollo de un modelo de deep learning para la colorización de bocetos</h1>

---

## Institución educativa:

- Universidad Privada Antenor Orrego

## Docentes:

- Cueva Chavez, Walter
- Lozano Chu, Ali

## Curso:

- Administración y arquitectura de mainframes 1

## Integrantes:

- Ferrer Deza, Nicolle Alejandra
- Velasquez Chorres, Axel Gabriel

## Descripción del caso de estudio

Este proyecto sienta sus bases en el aumento de criminalidad que se ha manifestado a lo largo de estos últimos años. Normalmente al ocurrir estos hechos, en muchos de los casos el delincuente es capaz de huir, aunque la víctima es capaz de recordar su cara y ciertas facciones de su cara, es en este momento en el que interviene un dibujante que trata de hacer un boceto hablado del delincuente. Entonces es en este momento en el que nosotros decidimos intervenir, si bien el boceto puede ayudar a darse una idea, puede no ser lo suficientemente realista como para realizar un aporte significativo, muchas veces esto es insuficiente. 

## Justificación

Nosotros como estudiantes de la carrera de ingeniería de sistemas buscamos realizar un aporte a esta situación, el cual consiste en utilizar un modelo de deep learning como lo son las GAN’s, usando como variable de entrada los sketch que pueden ser realizados y generar una imagen a color de un rostro realista, con el objetivo de que esta imagen generada pueda ser de mayor utilidad a la hora de cumplir con el objetivo de identificar a los responsables de estas actividades ilícitas.

## Objetivos

- Objetivo general:
  Realizar un proyecto de ciencia de datos cumpliendo el ciclo de vida de la metodología CRISP-DM
- Objetivos especificos:
  - Generar imagenes realistas a traves de bocetos
  - Realizar un análisis de la situación criminalisica en Perú
 
## Marco teórico

### 1. GAN

Las redes generativas adversarias se sustentan en el enfrentamiento de dos redes neuronales que compiten en un juego continuo de suma cero. Es decir, la pérdida o ganancia de una de esas redes se compensa con la pérdida o ganancia de la opuesta. 

![Arquitectura GAN](https://github.com/AVelasquezCH/Sketch2Picture/blob/master/Otros/Imagenes/GAN.jpg)

En la imagen anterior podemos apreciar la estructura de las GAN, las cuales cuentan con 2 elementos clave; el generador y el discriminador. El generador como su propio nombre nos indica, se encarga de crear las imágenes según la entrada dada; y el discriminador se encargar de evaluar que tan parecido es la imagen generada a la imagen objetivo. Por cada época, se compara la imagen generada con el objetivo, se determinan los costes y se procede a aplicar los gradientes, tanto al generador como al discriminador, con lo cual se produce el aprendizaje.

### 2. cGAN

Luego de entender que es una GAN, podemos definir la cGAn, que es el tipo específico que se procederá a aplicar en el proyecto. Una cGAN es una red generativa adversaria condicionada, en la cual se obtiene una salida la cual es condicionada por la entrada o etiqueta dada, es decir que la salida que se obtenga será relacionada a la entrada la cual le ha sido otorgada al modelo.

### 3. Adam

El nombre de Adam deriva de estimación del momento adaptativo, este es un algoritmo de optimización que puede ser usado en reemplazo del clásico descenso del gradiente estocástico que recientemente ha tenido una adopción más amplia para aplicaciones de deep learning en visión por computadora y procesamiento de lenguaje natural. El descenso del gradiente estocástico conserva una sola tasa de aprendizaje para todas las actualizaciones de peso y la tasa de aprendizaje no varía en el entrenamiento, mientras que Adam computa tasas de aprendizaje adaptativas individuales para diferentes parámetros a partir de estimaciones del primer y segundo momento del gradiente, esta combina las ventajas de otras 2 extensiones como lo son AdaGrad y RMSProp. 

- De AdaGrad mantiene una tasa de aprendizaje por parámetro que mejora el rendimiento en problemas con gradientes.
- De RMSProp mantiene tasas de aprendizaje por parámetro que se adaptan en función del promedio de magnitudes recientes de los gradientes para el peso; aunque en lugar de adaptar el parámetro de tasa de aprendizaje basado en el promedio del primer momento, Adam usa el promedio del segundo momento del gradiente.

El algoritmo calcula un promedio del primer y el segundo momento, y los parámetros beta1 y beta2 controlan la tasa de caída de estos promedios móviles.

### 4. U-nets

A partir de una red neuronal convolucional tradicional y después algunas modificaciones en la arquitectura se originó la U-net. El fin de utilizar este tipo de red, es porque puede localizar y distinguir bordes mediante una clasificación en cada píxel, por lo que la entrada y la salida comparten el mismo tamaño
La red a simple vista tiene la forma de “U”, la arquitectura simétrica que posee se divide en dos partes: la convolución  y la deconvolución.
Mediante la convolución , a medida que las imágenes pasan por cada capa van reduciendo su tamaño; a diferencia de la deconvolución que se encarga de aumentar el tamaño de la imagen hasta recuperar el tamaño original. La razón de usar estas dos rutas es combinar la información de las capas anteriores para obtener una predicción más precisa. El modelo de una U net puede ser graficado con la siguiente imagen.

![Unet](https://github.com/AVelasquezCH/Sketch2Color/blob/master/Otros/Imagenes/Unet.jpg)

## Procedimiento

Para el desarrollo de este proyecto, decidimos dividirlo en 2 grupos; una parte en la que se tiene como objetivo hacer un analisis de la realidad peruana sobre la criminalidad para tener una justificación para el desarrollo del modelo de deep learning, y la segunda etapa consistira en el desarrollo del modelo anteriormente mencionado. Para la primera parte del desarrollo del proyecto, se uso la plataforma cloud de Azure, la cual entre uno de sus muchos servicios, incluye el de creación y administración de maquinas virtuales. En esta, se monto una maquina que funciona con el sistema operativo de Debian en su versión 10. Se configuro conexión ssh para garantizar su seguridad, se instalo Jupyter Lab, se implemento maria DB con phpmyadmin y como lenguaje se utilizo python.
Para el desarrollo del modelo se uso principalmente TensorFlow y se monto toda la arquitectura sobre la plataforma de Google Colab, esto debido a que nos otorga la opción de elegir un tipo de entorno con GPU, lo cual sera necesario para el desarrollo del modelo. Como ya se explico anteriormente, se porcedera a desarrollar la metodología de ciencia de datos CRISP-DM, por lo que a continuacion se indicaran sus fases y como es que cada una de estas fue satisfecha.

### 1. Recolección de datos

Para el desarrollo de esta etapa, principalmente se usaron 4 datasets; eston son los encontrados en el [Sistema Integrado de Estadisticas de la Criminalidad y Seguridad Ciudadana](http://datacrim.inei.gob.pe/panel/mapa), el encontrado en [El proyecto de data criminal](https://inciudadana01.wixsite.com/datacriminal/base-de-datos) desarrollado por estudiantes de periodismo y los dataset de CUHK y AR que pueden ser encontrados [aquí](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html), los cuales contienes 2 grupos de imagenes, el grupo de sketch, que servira como input, y el grupo que contiene las imagenes de las personas reales, las cuales serviran como objetivo para el desarrollo del modelo de deep learning en la etapa posterior.

Los datasets utilizados para el analisis son los dos primeros; para el primero de estos se realizo la tecnica de web scrapping y luego de almaceno en un formato csv, mientras que el segundo dataset consistia en un excel, por lo que simplemente se procedio a ordenarlo y añadirlo a un csv.

### 2. Preparación de datos

Para esta etapa, los dos datasets que contenian información de la criminalidad fueron sometidos a funciones en las cuales se limpió la información que contenian. Al primer dataset se le procedió a extraer solo el texto en algunos campos que contenían caracteres extraños y eliminar espacios al principio y al final de las cadenas de texto; luego de realizar esta limpieza se procedió a almacenar la información ya limpia en un nuevo csv. Para el segundo dataset se adapto al fecha al formato aceptado por la base de datos, se llenaron datos nulos, se eliminaron datos anomalos, para ciertos campos se eliminaron los espacios al inicio y al final, y se pusieron en mayúsculas algunos campos que contenían cadenas de caracteres.

### 3. Modelo de datos estructurados

Para el modelado de datos estructuradas se consideraron 9 tablas, las cuales pueden ser identificadas en la siguiente imagen.

![BD](https://github.com/AVelasquezCH/Sketch2Color/blob/master/Otros/Imagenes/BD.png)

Como se puede apreciar, hay 2 tablas principales las cuales se insertaron la mayoria de datos relevantes para el analisis a realizar, las cuales son Crimenes_DataCrim y Robos_Lima. Basicamente el nombre de estas indica el respectivo dataset del que se obtuvieron, y seran de las más relevantes para las siguientes etapas.

### 4. Transformación y consultas exploratorias

Para esta etapa, se realizaron 10 consultas exploratorias con el objetivo de analizar la data obtenida y ayudar a esclarecer la situación nacional que se tiene respecto al indice de criminalidad segun los datos que se incluyen en los datasets. Algunas de estas consultas se usaran en la siguiente etapa, debido a que se definio que podria ser importante graficarlas para continuar con el proceso de analisis que buscamos realizar en este estudio.

### 5.Exploración visual de los datos

En esta etapa se realizan igualmente consultas a la bd para obtener información que se considera relevante para el analisis, aunque esta vez de manera gráfica. Como se menciono anteriormente, se reutilizaron algunas consultas anteriores porque se considero que era relevante graficarlas, y tambien se realizaron 10 de estos graficos los cuales son mostrados en un dashboard.

### 6. Modelo

Para el desarrollo del modelo se utilizaron los 2 datasets de imagenes mencionados anteriormente, en primera instancia se procedio a agrupar ambos en los respectivos grupos de sketch y las imagenes de objetivo, que serian las imagenes de las personas a color. Luego de que se tuvo las imagenes ordenadas en sus respectivas carpetas, se subio a la plataforma de Google Drive, las cuales posteriormente seran referenciadas en Google Colab para la implementación del modelo

Para la realización de este modelo nos basamos en las fuentes publicas del modelo [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix), las fuentes de un modelo de [CycleGan](https://www.tensorflow.org/tutorials/generative/cyclegan) y dos articulos encontrados en [esta página](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/) y en [esta otra](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/). Estas fuentes sirvieron como base para el inicio de desarrollo; los articulos fueron utilizados para comprender la teoria y conceptos necesarios para el desarrollo del proyecto, y los tutoriales de tensorflow se usaron como guia para el desarrollo del modelo.

Para el proyecto se utilizaron paquetes como lo son tensorflow, keras, matplotlib, entre otras. De los modelos de guía se extrajo la clase de random jitter, la cual ayuda a que el modelo optimice la generación de imagenes, aunque se le realizo una pequeña variación a este metodo, con la cual se obtuvieron mejores resultados. Para la carga de imagenes no se necesito realizar ningun preprocesamiento, excepto el de el escalamiento de estas al rango de -1 a 1, tal como si se les hubiera aplicado una tanh. luego de esto se definio la función de downscaling, la cual consta de una funcion de Conv2D y una función de LeakyRelu(); mientras que la función de upscaling contiene una Conv2DTranspose, un Dropout del 0.5 y una capa de activación Relu, a excepción de la ultima capa, en la cual solo se incluye la Conv2DTranspose.

Luego de tener definida las funciones se procede a crear la funcion que definira al generador, para este se definio un arreglo con las capas respectivas para el downscaling y el upscaling, se recorre capa por capa, se hacen las skip connections y se retorna la entrada y la salida, el grafo del generador puede ser visto en la siguiente imagen.

![Generador](https://github.com/AVelasquezCH/Sketch2Color/blob/master/Otros/Imagenes/Generador.png)

Pasa lo mismo con la función que define el discriminador, aunque este solo hace un downscaling, teniendo como input la imagen generada por el modelo y la imagen de entrada, en la siguiente imagen se tiene el grafo generado por el discriminador.

![Discriminador](https://github.com/AVelasquezCH/Sketch2Color/blob/master/Otros/Imagenes/Discriminador.png)

Luego se proceden a definir los casos de coste; para el coste del discriminador se calculara primero su costo al analizar la imagen objetivo, luego se calcula el costo del discriminador al analizar la imagen generada, ambos resultados se suman y se retorna este valor. Para el costo del generador se calcula primero su costo al analizar la imagen que ha generado, luego se calcula la media de la imagen objetivo versus la imagen generada, y luego se suma el costo de la imagen generado al promedio del valor anteriormente calculado multiplicado por un valor lambda, esto con el objetivo de mejorar el aprendizaje; para este modelo se determino que con el valor lambda de 100 se obtenian mejores resultados. 

Posterior a la definicion de las funciones de costo, se define la funcion generar, a la cual se le brindara el input, este se pasara al modelo para que realice su proceso, se recive el resultado, se almacena como una imagen y se muestra como output. La ultima de las funciones a definir es la que se encragara del proceso de entrenamiento, en esta se recibiran un numero de epocas, y por cada una de estas se recorreran las imagenes de entrenamiento. Se calculara los costes luego de obtener la imagen generada y se aplicaran los gradientes tanto al generador como al discriminador, con lo cual se llevara a cabo el aprendizaje. Luego de esto, tambien en cada epoca, luego de recorridas las imagenes de entrenamiento, se recorren las imagenes de validación se le envia al generador y se muestra como salida.

Luego de definidas las funciones, se cargan las imagenes y se dividen las particiones de prueba y de validación, y se almacenan en tensores; también se definen los objetos que representaran al generador y al discriminador, y se definen las funciones de costes. Para desarrollar este modelo, se utilizo a Adam como función que determine el coste, ya que ha demostrado que tiene buenos resultados con este tipo de modelos, a este se le definio como parametros una taza de aprendizaje de 2e-4 y un beta_1 de 0.5, debido a que en todas las pruebas, estos valores arrojaron los mejores resultados.

Luego de tener todas las funciones anteriores definidas, se procede a entrenar el modelo. Para el entrenamiento, se determino que con un numero de 150 epocas se alcanzaba un resultado aceptable.

### 7. Exportación y Comunicación

Para esta etapa, se realizaron videos que demuestran la evaluacion en cada epoca para algunas imagenes de validación, y tambien se almacenan en Google Drive las imagenes que se generan al realizar una validación con cualquier imagen.

## Conclusiones

El proyecto cumple con el objetivo de generar las imagenes con el dataset de entrenamiento y con la partición de validación con un costo del generador que varia de 10 a 5; sin embargo, al realizar la validación de imagenes, denoto que cuando se le brinda imagenes que no ha visto, en una minoria de estas se producen errores; y si bien cumple en ciertos rasgos, aun presenta ciertos fallos. Esto podria deberse a que en algunas de las imagenes usadas para la validación, estas tienen un estilo de dibujo diferente y esto puede estar genereando algún conflicto. Podría decirse que el modelo cumplio de una manera exitosa la generación de rostros realistas a color con el dataset de entrenamiento y que para elementos de validación que posean un estilo de boceto similar al que se le brindo de entrenamiento produce imagenes que cumplen con el objetivo, pero que podría requerir un mayor dataset para el entrenamiento con mas estilos de boceto, con lo cual podria llegar a ser capaz de colorear cualquier imagen bocetada que se le presente.

Ademas de lo descrito, tambien podemos concluir que gracias al analisis realizado en la primera etapa del proyecto hemos podido determinar basado en hechos que el nivel de criminalidad en nuestro país ha ido aumentando exponencialmente. Gracias a los graficos y las consultas realizadas, pudimos obtener más información que ayudo a entender los patrones o los distintos grupos donde estos delicuentes más atacan; entre alguno de estos se encuentra que el la ciudad más atacada por la delincuencia en nuestro país es la ciudad de Lima y su distrito más atacado es San Juan de Lurigancho, también que el mayor numero de robos segun la data analizada se realiza a empresarios, entre otros datos. Este analisis nos sirvio para esclarecer que siendo el Perú un país tan atacado por la delincuencia, puede que requiera un modelo como el que se implemento, con el objetivo de intentar disminuir este tipo de incidencias.
