<h1 align="center">Proyecto Sketch to Picture</h1>

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

Este proyecto sienta sus bases en el aumento de criminalidad que se ha manifestado a lo largo de estos ultimos años. Normalmente al ocurrir estos hechos, en muchos de los casos el delincuente es capaz de huir, aunque la victima es capaz de recordar su cara y ciertas facciones de su cara, es en este momento en el que interviente un dibujante que trata de hacer un boceto habla del delincuente. Entonces es en este momento en el que nosotros decidimos intervenir, si bien el boceto puede ayudar a darse una idea, puede no ser lo suficientemente realista como para realizar un aporte significativo, muchas veces esto es insuficiente. 

Es por esto que nosotros como estudiantes de la carrera de ingeniería de sistemas buscamos realizar un aporte a esta situación, y nuestro aporte consiste en utilozar un modelo de deep learning como lo son las GAN, con el objetivo de tener como variable de entrada los sketch que pueden ser realizados y generar una imagen a color de un rostro realista, con el objetivo de que esta imagen generada pueda ser de mayor utilidad a la hora de cumplir con el objetivo de identificar a los responsables de estas actividades ilicitas.

## Procedimiento

Para el desarrollo del proyecto, principalmente se usaron 4 datasets; y el proyecto de ha dividido en 2 partes, la primera que se base en una recopilación de datos criminales que han sido encontrados y que se centran en crimenes desarrollados en Perú, eston son los encontrados en el [Sistema Integrado de Estadisticas de la Criminalidad y Seguridad Ciudadana](http://datacrim.inei.gob.pe/panel/mapa) y el encontrado en [El proyecto de data criminal](https://inciudadana01.wixsite.com/datacriminal/base-de-datos) desarrollado por estudiantes de periodismo. Con estos datasets pudimos desarrollar la parte previa al desarrollo del modelo, el cual consiste en el analisis y la exploración de toda esta data, a fin de desarrollar una justificación del proyecto.

Para la segunda parte y el desarrollo del modelo, usamos los dataset de CUHK y AR que pueden ser encontrados [aquí](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html), los cuales contienes 2 grupos de imagenes, el grupo de sketch, que servira como input, y el grupo que contiene las imagenes de las personas reales, las cuales serviran como objetivo, con lo cual se entrenara al modelo de deep learning.

Para la primera parte del desarrollo del proyecto, se uso la plataforma cloud de [Azure](https://azure.microsoft.com/en-us/free/search/?&ef_id=CjwKCAjw1K75BRAEEiwAd41h1H9usN3bMFuz-8kDhYj0OVgOODf7i6OnaA5J8SDJwyxGyolhJbOMqRoCq08QAvD_BwE:G:s&OCID=AID2100093_SEM_CjwKCAjw1K75BRAEEiwAd41h1H9usN3bMFuz-8kDhYj0OVgOODf7i6OnaA5J8SDJwyxGyolhJbOMqRoCq08QAvD_BwE:G:s&dclid=CjgKEAjw1K75BRCyrLr7jv_W8lcSJADL7AOmrzE8SNawdyvgVxGHbO-ERpRJC89IXTuWOwj31nLXa_D_BwE), la cual entre uno de sus muchos servicios, incluye el de creación y administración de maquinas virtuales. En esta, se monto una maquina que funciona con el sistema operativo de Debian en su versión 10. Se configuro conexión ssh para garantizar su seguridad, para la parte de manipulacion, recopilacion y exploración se utilizo la Notebook de Python con diversos paquetes, y para la base de datos se implemento maria DB con phpmyadmin.

Para el desarrollo del modelo, decidimos utilizar las redes generativas adversarias o GAN por sus siglas en inglés. Decidimos usar esto debido a que a lo estudiado anteriormente en el curso, puede ser lo más apto con el objetivo de el generar imagenes. Para el desarrollo de lo mismo, se uso principalmente [TensorFlow](https://www.tensorflow.org/) y se monto toda la arquitectura sobre la plataforma de [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb),  debido a que nos otorga la opción de elegir un tipo de entorno con GPU, lo cual sera necesario para el desarrollo del modelo.

![Arquitectura GAN](img/GAN.png)

En la imagen anterior podemos apreciar la estructura de las GAN, las cuales cuentan con 2 elementos clave; el generador y el discriminador. El generador como su propio nombre nos indica, se encarga de crear las imagenes segun la entrada dada; y el discriminador se encargar de evaluar que tan parecido es la imagen generada a la imagen objetivo. Por cada época, se compara la inagén generada con la objetivo, se determinan los costes y se procede a aplicar los gradientes, tanto al generador como al discriminador, con lo cual se produce el aprendizaje.

## Conclusiones

