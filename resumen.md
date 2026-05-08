# Resumen de la Materia PLN

## ¿Qué es transfer Learning?

El Transfer Learning (o aprendizaje por transferencia) es una técnica en la que un modelo desarrollado y entrenado para una tarea general se reutiliza como punto de partida para entrenar un modelo en una segunda tarea más específica. Básicamente, consiste en aprovechar el conocimiento que el modelo ya adquirió previamente (usando grandes cantidades de datos) para resolver problemas nuevos.

## ¿Cómo funciona?

En lugar de que el modelo empiece a aprender desde cero mediante ensayo y error (con una inicialización aleatoria), el entrenamiento arranca desde un punto avanzado donde los "pesos" del modelo ya están cerca de lo óptimo y conocen patrones fundamentales. Por lo general, las primeras capas de la red neuronal mantienen las características generales que ya aprendieron (como la sintaxis de un idioma), mientras que las capas más profundas se ajustan y especializan en la nueva tarea (por ejemplo, entender la semántica particular de un texto legal).

## ¿Cuáles son sus ventajas principales?

- Ahorro de recursos y tiempo: Al no tener que aprender todo desde cero, se reduce drásticamente el tiempo de entrenamiento y la capacidad de cómputo necesaria.
- Menor necesidad de datos: Permite lograr buenos resultados incluso cuando se tienen datos muy limitados para la nueva tarea específica.
- Mejor desempeño: Facilita que los modelos se adapten y generalicen mejor con solo realizarles pequeños ajustes.

Ejemplos clásicos: En procesamiento de lenguaje natural (NLP), esto se ve cuando tomamos modelos preentrenados masivamente (como Word2Vec o BERT) y los usamos como base para tareas concretas, como crear un clasificador de reseñas de hoteles. En visión por computadora, equivale a usar modelos preentrenados con miles de imágenes generales (como ImageNet) para luego hacer clasificación de imágenes médicas muy específicas.

## ¿Qué es la semántica distribucional?

La semántica distribucional es una teoría que establece que el significado de una palabra está determinado por las palabras que frecuentemente la rodean (su contexto). Se resume perfectamente en una famosa frase del lingüista J. R. Firth: "Conocerás a una palabra por la compañía que frecuenta". En la práctica, esto significa que para que una máquina entienda una palabra, analiza grandes cantidades de texto y recolecta qué otras palabras aparecen cerca de ella (por ejemplo, las 5 palabras anteriores y las 5 posteriores). Un ejemplo muy claro: Imagina que lees un texto sobre un objeto inventado llamado "galorim" y ves oraciones como: "el tic-tac del galorim" , "ajustamos el galorim por la zona horaria" o "el galorim me avisa si me despierto tarde". Aunque nunca antes hayas visto esa palabra, su contexto te permite deducir instantáneamente que se trata de un reloj o despertador. ¡Esa es la esencia de la semántica distribucional!

## ¿Por que son importantes los embeddings?

Los embeddings son fundamentales en el Procesamiento de Lenguaje Natural (NLP) moderno por las siguientes razones clave:

- Traducen el lenguaje para la máquina: Las redes neuronales requieren entradas numéricas para poder funcionar. Los embeddings resuelven esto convirtiendo las entidades (como las palabras) en vectores (listas de números densas) que la computadora puede procesar de manera natural.
- Capturan el significado real (semántica): No asignan números al azar. Su gran valor es que logran capturar el significado profundo de las palabras y las relaciones sintácticas o semánticas entre ellas. En este formato matemático, las palabras con significados similares terminan agrupadas muy cerca unas de otras.
- Son compactos y eficientes: Métodos antiguos (como el one-hot encoding) usaban vectores gigantes y poco eficientes, llenos de ceros. Los embeddings, en cambio, comprimen toda esa información en dimensiones reducidas (por ejemplo, 300 dimensiones), lo que ocupa menos espacio y facilita el cálculo.
- Garantizan el éxito del modelo: En Deep Learning, la regla básica es que "mejores representaciones llevan a un mejor rendimiento". Usar buenos embeddings es la pieza clave para que los modelos logren alta precisión en tareas como clasificación de textos, búsquedas o análisis semántico.
## ¿Qué es un embedding?

Un embedding es una representación matemática de una entidad (como puede ser una palabra, una imagen, un audio o un documento) en forma de un vector denso, es decir, una lista compacta de números. Básicamente, como las redes neuronales y las computadoras no entienden el texto o las imágenes directamente, el modelo de embedding se encarga de "traducir" estas entidades a un formato numérico que sí pueden procesar. Lo más valioso de un embedding es que no asigna números al azar, sino que aprende a capturar los aspectos más importantes y el significado de esa entidad. Por ejemplo, en el caso de las palabras, un buen embedding captura su semántica, logrando que palabras con significados parecidos (como "rey" y "reina") terminen siendo representadas por vectores que están matemáticamente agrupados "cerca" en el espacio. Además, al ser vectores de tamaño fijo y reducido (por ejemplo, de 300 dimensiones), resultan mucho más eficientes de almacenar y calcular que otros métodos más antiguos.

## ¿Qué son embeddings contextuales?

Los embeddings contextuales son representaciones numéricas dinámicas de las palabras, donde el vector que se le asigna a una palabra no es fijo, sino que depende específicamente del contexto (la oración completa) en el que se encuentra. A diferencia de los embeddings estáticos clásicos (como Word2Vec o GloVe) que tienen un único vector de diccionario para cada palabra, los embeddings contextuales solucionan el problema de la polisemia (palabras con múltiples significados). Un ejemplo claro: Si usamos la palabra "banco", un modelo de embeddings contextuales le asignará un vector matemático diferente si la oración es "Saqué dinero del banco" (institución financiera) que si la oración es "Me siento en el banco" (asiento). Para lograr esta flexibilidad, no se basan solo en tablas de conteo, sino que se generan utilizando arquitecturas de redes neuronales más avanzadas que procesan y comprenden oraciones completas. Algunos de los modelos más conocidos que utilizan embeddings contextuales son ELMo, BERT y GPT.

## ¿Cómo se implementa el transfer learning en una red neuronal?

Se implementa principalmente en dos grandes pasos:

1. Pre-training (Preentrenamiento): Se entrena un modelo inicial utilizando una enorme cantidad de datos generales (usualmente sin etiquetar) en una tarea pretexto. El objetivo es que la red aprenda los patrones generales del lenguaje, logrando que sus "pesos" o parámetros queden inicializados en un punto muy avanzado, mucho mejor que arrancar desde valores aleatorios.
2. Fine-tuning (Ajuste fino): Se toma ese modelo preentrenado y se lo sigue entrenando (se ajustan sus pesos) pero ahora usando los datos específicos (y etiquetados) de la tarea final que queremos resolver. En este proceso, las primeras capas de la red suelen conservar las características generales aprendidas (como relaciones sintácticas), mientras que las capas más profundas se especializan en los detalles de la nueva tarea.
## ¿Cuáles son las ventajas del transfer learning en términos de los datos de entrenamiento?

En términos de datos de entrenamiento, el transfer learning ofrece las siguientes ventajas principales:

- Reduce la cantidad de datos necesarios: Permite lograr buenos resultados en tareas específicas incluso cuando se tienen datos muy limitados para entrenar,,.
- Menor dependencia de etiquetas manuales: Disminuye drásticamente la necesidad de usar datos etiquetados por humanos, lo cual es vital en áreas donde anotar datos es costoso o difícil (como en diagnósticos médicos),.
- Aprovecha datos no etiquetados: Permite sacar provecho de cantidades masivas de datos generales y sin etiquetar durante su etapa inicial para aprender los patrones básicos.
## ¿Qué interpretación tiene que dos palabras se encuentren "cerca" en un espacio vectorial?

Que dos palabras se encuentren "cerca" en un espacio vectorial significa fundamentalmente que tienen significados similares o que suelen utilizarse en contextos parecidos. En los modelos de word embeddings, el vocabulario se traduce a un espacio matemático continuo donde cada palabra es un punto (un vector). El modelo aprende automáticamente a agrupar en la misma zona de ese espacio a las palabras que comparten características semánticas. Por ejemplo, palabras como "rápido" y "veloz" (sinónimos), o "rey" y "reina" (conceptos muy relacionados), tendrán valores matemáticos que las ubicarán muy cerca la una de la otra, mientras que palabras sin ninguna relación estarán muy separadas

## Describa el método Skip-Gram en W2V

El método Skip-Gram es una de las arquitecturas principales del modelo Word2Vec. Su funcionamiento se basa en tomar una palabra central específica y tratar de predecir cuáles son las palabras que la rodean (su contexto). Para ir al grano, el proceso se resume en cuatro pasos:

1.
2.
3.
4. Comienza asignando valores matemáticos aleatorios.

Recorre (itera) una a una cada palabra del texto. Intenta predecir las palabras vecinas dentro de una ventana de contexto específica dada esa palabra central. Ajusta sus valores (minimiza la función de pérdida) para mejorar sus predicciones paso a paso. Además, como calcular estas probabilidades para todo el vocabulario es computacionalmente muy costoso, la arquitectura suele utilizar una técnica llamada negative sampling para agilizar el proceso

## Describa el método Glove?

El método GloVe (Global Vectors) es un algoritmo para crear embeddings de palabras que se caracteriza por basarse en el contexto global de todo el texto, a diferencia de Word2Vec que se enfoca en ventanas de contexto locales (pequeñas). Para ir al grano, su proceso se resume en los siguientes puntos:

1. Matriz de co-ocurrencias: En lugar de predecir palabra por palabra, GloVe construye una tabla estadística (matriz) gigante donde cuenta cuántas veces cada palabra del vocabulario aparece junto a cualquier otra palabra en todo el corpus de texto.
2. Análisis de proporciones (Ratios): El modelo no usa el conteo o la probabilidad bruta, sino que analiza las proporciones de cómo cambian esas probabilidades de co- ocurrencia entre distintas palabras. Esto le permite capturar diferencias muy sutiles de significado y relaciones semánticas complejas.
3. Función de ponderación: Aplica una fórmula matemática para evitar que las palabras extremadamente frecuentes (como "el", "de", "y") dominen el entrenamiento, y al mismo tiempo le quita peso a las palabras demasiado raras.
4. Vectores finales: Al reducir matemáticamente esta matriz gigante, el modelo genera los vectores finales, logrando que las palabras que co-aparecen frecuentemente en contextos similares terminen muy juntas en el espacio vectorial
## ¿Cual es la diferencia conceptual entre word2vec y Glove?

La diferencia conceptual clave radica en el alcance del contexto que utilizan para aprender:

- Word2Vec (Contexto Local): Es un modelo predictivo que analiza el texto enfocándose en ventanas pequeñas (por ejemplo, el entorno de 5 palabras). Aprende paso a paso intentando predecir una palabra a partir de sus vecinas, o viceversa.
- GloVe (Contexto Global): Es un modelo que aprovecha la información estadística de todo el corpus de texto a la vez. Construye una matriz gigante que cuenta las co- ocurrencias de todas las palabras a nivel global y luego aprende analizando las proporciones (ratios) de esas probabilidades conjuntas.

Word2Vec aprende "prediciendo" sobre pedacitos cortos de texto, mientras que GloVe aprende extrayendo información de una tabla estadística que abarca la totalidad del texto

## ¿Qué es la tokenización?

La tokenización es el proceso de dividir o segmentar un texto en unidades más pequeñas y fundamentales (llamadas "tokens") para que los modelos de lenguaje puedan trabajar con ellas. Un token no siempre equivale a una palabra entera; dependiendo del método, puede ser una sola letra, un número, un signo de puntuación o un fragmento de palabra. Por ejemplo, la expresión "¡Decímelo!" podría dividirse en una lista de múltiples tokens: ["¡", "Decí", "me", "lo", "!"]. Existen distintas formas de hacerlo:

- Métodos clásicos: Simplemente cortan el texto guiándose por los espacios o signos de puntuación, o dividen todo letra por letra.
- Subwords (Sub-palabras): Es el método que usan los modelos modernos (como

BPE en GPT o WordPiece en BERT). Busca un equilibrio dividiendo las palabras raras o desconocidas en fragmentos más pequeños y frecuentes, lo que evita que el modelo colapse ante palabras nuevas y reduce enormemente el tamaño del diccionario que necesita memorizar

## ¿Qué es un modelo de lenguaje?

Un modelo de lenguaje es, fundamentalmente, un sistema diseñado para predecir la probabilidad de una secuencia de palabras. Para ir al grano, funciona así: el modelo analiza cantidades masivas de datos de texto para aprender los patrones, reglas y estructuras subyacentes de un idioma. Gracias a todo ese conocimiento previo, el modelo es capaz de:

1. Predecir la siguiente palabra: Adivinar qué palabra tiene más sentido estadístico que siga a continuación en una frase (por ejemplo, si lee "Voy a guardar el queso en el...", predecirá con alta probabilidad la palabra "refrigerador").
2. Evaluar oraciones: Calcular qué tan probable o natural es una oración o texto completo dentro de ese idioma
## Que es un modelo de lenguaje unidireccional?

Un modelo de lenguaje unidireccional es aquel que procesa el texto en una sola dirección, lo que significa que para entender o predecir una palabra solo puede mirar hacia el pasado (su contexto previo). Sus características principales son:

- Procesa de izquierda a derecha: A diferencia de los modelos bidireccionales que miran toda la oración junta, el unidireccional no puede "ver" las palabras que están a la derecha de la palabra actual.
- Cálculo de probabilidad: Matemáticamente, calcula la probabilidad de una oración completa multiplicando la probabilidad de cada palabra condicionada únicamente por todas las palabras que la precedieron ().
- Ejemplo clásico: Los modelos generativos que utilizan una arquitectura "solo decodificador" (decoder-only), como la familia GPT, son unidireccionales y se especializan en generar texto nuevo palabra por palabra
## ¿Qué es un modelo de lenguaje generativo?

Un modelo de lenguaje generativo es un sistema capaz de producir secuencias de palabras o texto nuevo basándose en los patrones estadísticos que aprendió al analizar grandes cantidades de datos. Funciona prediciendo secuencialmente qué palabra es la más probable que siga a continuación. Por ejemplo, si el modelo procesa la frase "Voy a guardar el queso en el...", utilizará su conocimiento para generar la palabra "refrigerador", y a partir de ahí continuará agregando palabras paso a paso para completar la idea

## Explique el proceso autoregresivo de los modelos generativos

El proceso autoregresivo es el mecanismo secuencial mediante el cual los modelos generativos producen texto nuevo paso a paso, prediciendo siempre la siguiente palabra basándose en las palabras anteriores (su historia o contexto). Para ir al grano, funciona así:

1.
2.
3. El modelo analiza una secuencia de entrada y, usando su conocimiento previo, calcula

matemáticamente qué palabra tiene más probabilidades de continuar la frase. Una vez que genera esa nueva palabra, la añade a la secuencia original. Luego, utiliza esa nueva secuencia (ahora más larga) como entrada para predecir la palabra que le sigue, y repite este ciclo palabra por palabra. Un ejemplo paso a paso ilustra muy bien esto: si el modelo procesa "Voy a guardar el queso en el", genera "refrigerador". Luego, toma la frase completa "Voy a guardar el queso en el refrigerador" como nuevo contexto para generar la palabra "para", y así sucesivamente hasta formar "Voy a guardar el queso en el refrigerador para conservarlo fresco"

## ¿Qué es un modelo de lenguaje n-gram?

Un modelo de lenguaje n-gram es un método estadístico clásico que predice cuál es la siguiente palabra en una secuencia basándose únicamente en las últimas palabras anteriores (una ventana de tamaño fijo ), en lugar de mirar toda la oración completa. Para ir al grano, sus características son:

- Contexto limitado: En lugar de buscar y contar una oración entera en sus datos de entrenamiento (lo cual es muy difícil porque el lenguaje es creativo y genera frases únicas), recorta el contexto para analizar solo un pedacito.
- Tipos comunes: Si se llama Unigrama (analiza palabras sueltas como "gato"); si es un Bigrama (analiza pares consecutivos como "comer al"); si es un Trigrama, y así sucesivamente.
- Objetivo: Utiliza el conteo de cuántas veces apareció ese pequeño "n-grama" en los datos para adivinar qué palabra tiene más sentido que siga a continuación
## ¿Cómo se entrena un modelo n-gram?

El entrenamiento de un modelo n-gram, a diferencia de las redes neuronales más complejas, no funciona por "ensayo y error", sino que se basa fundamentalmente en contar palabras y calcular proporciones matemáticas. Los pasos para "cocinar" o entrenar este modelo son:

1.
2.
3.
4. Recopilación y limpieza: Se toma un corpus de texto enorme (que represente bien al

lenguaje), se pasa todo a minúsculas y se limpia de caracteres no deseados. Tokenización y creación de n-gramas: Se divide el texto en palabras individuales y luego se arman las secuencias del tamaño elegido (por ejemplo, si , se arman todos los pares de palabras posibles). Establecer el vocabulario: Se guardan estas combinaciones y secuencias basándose en su frecuencia de aparición en el texto. Cálculo de probabilidades (Estimador de Máxima Verosimilitud): Aquí ocurre el entrenamiento en sí. El modelo cuenta cuántas veces aparece una secuencia exacta. Para saber qué probabilidad hay de que siga una palabra específica, simplemente divide la cantidad de veces que apareció la frase completa por la cantidad de veces que apareció solo el contexto previo. En resumen: entrena leyendo mucho texto, cortándolo en pedacitos de tamaño fijo y memorizando estadísticamente cuántas veces apareció cada pedacito para adivinar qué sigue después.

## ¿Cuáles son las principales desventajas de un modelo ngram?

Las principales desventajas de los modelos n-gram se pueden resumir en tres puntos clave:

- Memoria corta (Pérdida de contexto): Como el modelo solo toma decisiones basándose en una ventana fija de las últimas palabras, no puede "ver" más allá de ese límite. Esto provoca una pérdida del contexto global, por lo que es incapaz de capturar relaciones o dependencias a largo plazo en una oración.
- Incapacidad de generalizar (Falta de datos): El lenguaje humano es muy creativo.

Es altamente probable que el modelo se encuentre con una secuencia de palabras que jamás vio en su texto de entrenamiento (lo que le daría probabilidad cero). Además, a diferencia de los embeddings, un modelo n-gram tradicional no entiende de semántica, por lo que no puede generalizar lo que aprendió hacia palabras similares o sinónimos.

- Problemas de almacenamiento (Escalabilidad): Para funcionar, el modelo necesita guardar en memoria todas las combinaciones (subcadenas) de palabras que existen en el corpus y contar su frecuencia. A medida que aumentamos el tamaño de para intentar darle más contexto al modelo, la cantidad de combinaciones posibles explota, volviéndose computacionalmente insostenible.
## ¿Cuál es la importancia del vocabulario en un modelo de lenguaje?

El vocabulario es el conjunto de palabras o tokens que un modelo de lenguaje es capaz de reconocer y generar. Su importancia radica en los siguientes puntos clave:

- Define los límites del modelo: Determina cuál es el espacio de búsqueda exacto con el que el sistema puede trabajar. Todo lo que quede fuera de este conjunto será procesado como desconocido.
- Afecta el rendimiento: Influye directamente en la precisión y en la capacidad de generalización del modelo. Contar con un buen vocabulario es fundamental para capturar los patrones lingüísticos de manera efectiva.
- Implica un costo computacional: Existe un equilibrio a tener en cuenta, ya que vocabularios más grandes permiten capturar más información, pero a costa de aumentar enormemente la complejidad computacional para el modelo
## ¿Qué es la lematización?

La lematización es el proceso de reducir las palabras a su forma base o raíz (llamada "lema"). Por ejemplo, transforma variaciones como "corriendo" en "correr", "anduvieron" en "andar" o "fueron" en "ser". Su utilidad se resume en estos puntos:

- Simplifica el texto: Reduce la inmensa variabilidad del lenguaje humano, permitiendo que el modelo trate palabras con formas diferentes (pero significados similares) como si fueran exactamente la misma entidad.
- Mejora la eficiencia: Al agrupar estas variaciones, disminuye drásticamente el tamaño del vocabulario que el sistema debe procesar y almacenar.
- Ayuda a generalizar: Es muy valiosa en modelos de lenguaje clásicos (como los n- gramas) para evitar memorizar redundancias y mejorar la capacidad de deducción del modelo
## ¿Qué ventajas trae la lematización?

Las principales ventajas de la lematización son:

- Simplifica el texto y unifica el significado: Reduce la gran variabilidad del lenguaje, permitiendo que el modelo trate palabras con formas diferentes (pero con el mismo significado base) como si fueran una sola entidad.
- Aumenta la eficiencia: Gracias a la unificación de términos, logra reducir el tamaño total del vocabulario que el sistema debe procesar, lo que aumenta notablemente la eficiencia del modelo.
- Mejora la generalización: Es especialmente útil para ayudar a modelos clásicos (como los n-gramas) a generalizar mejor lo aprendido, evitando que memoricen redundancias innecesarias en el vocabulario

Cuando conviene lematizar y cuando no, para que tareas? La decisión de lematizar o no depende fuertemente del tipo de modelo que estés utilizando y del objetivo final de tu tarea:

## Cuándo conviene lematizar:

- En modelos de lenguaje clásicos (como los n-gramas): Es sumamente útil porque agrupa palabras con formas diferentes pero significados similares bajo una misma entidad. Al reducir el tamaño del vocabulario, el modelo se vuelve más eficiente, generaliza mejor la información y no pierde memoria en redundancias gramaticales innecesarias.
- En tareas de análisis y extracción de características básicas: Cuando te interesa capturar el sentido central de un texto (por ejemplo, al contar frecuencias de palabras), llevar variaciones como "corriendo" o "anduvieron" a sus raíces "correr" o "andar" simplifica enormemente los datos.

Cuándo NO conviene lematizar:

- En tareas de generación de texto: Para generar oraciones que un humano pueda leer de forma natural, el vocabulario del modelo debe estar intacto y ser "completo". Si lematizas los datos, el modelo perdería los tiempos verbales o plurales y terminaría generando oraciones compuestas únicamente por verbos en infinitivo y palabras en singular.
- Al usar modelos modernos de Deep Learning (Transformers, BERT, GPT): En la actualidad, la tendencia con las redes neuronales es preservarlo todo. En lugar de lematizar, estos modelos manejan la variabilidad utilizando algoritmos de tokenización por sub-palabras (como BPE o WordPiece) o métodos como FastText que analizan fragmentos de caracteres. Esto les permite procesar raíces, prefijos y sufijos automáticamente, entendiendo la morfología rica del lenguaje sin perder los sutiles matices y detalles del contexto original
## ¿Qué es Precision, Recall y F1?

Estas son tres métricas fundamentales utilizadas para evaluar el rendimiento de los modelos:

- Precisión (Precision): Responde a la pregunta ¿Cuántas de las clasificaciones positivas que hizo el modelo son realmente correctas?. Se enfoca en la calidad de lo que el modelo identificó como positivo.
- Sensibilidad (Recall): Responde a la pregunta ¿Cuántos de los positivos reales que existían en los datos lograron ser detectados por el modelo?. Se enfoca en la capacidad del modelo para no "comerse" o perder casos positivos reales.
- F1 (F1 Score): Es una métrica general que calcula un balance o equilibrio matemático entre la Precisión y la Sensibilidad
## Mencione como ejemplo una tarea en el que sea conveniente maximizar

Precision y otra en la que sea conveniente maximizar el Recall. Aunque las fuentes proporcionadas definen matemáticamente estas métricas pero no listan ejemplos concretos de cuándo maximizar cada una, te ofrezco dos ejemplos clásicos (externos a tus apuntes) que ilustran perfectamente el concepto:

- Maximizar Precisión: Es conveniente en un filtro de correo spam. Aquí queremos minimizar los falsos positivos; es decir, queremos estar muy seguros de que lo que se clasifica como spam realmente lo sea, para evitar borrar por error un correo importante de trabajo.
- Maximizar Sensibilidad (Recall): Es conveniente en un modelo de diagnóstico médico (un campo mencionado en las fuentes). Aquí buscamos minimizar los falsos negativos; es decir, es vital detectar a todos los pacientes que puedan tener una enfermedad severa, incluso si el modelo genera algunas falsas alarmas (falsos positivos) que luego un médico deba descartar.
## ¿Cómo se evalúan los modelos de lenguaje?

La evaluación de los modelos de lenguaje busca determinar su calidad y eficacia, y se divide principalmente en dos grandes enfoques:

1. Evaluación Intrínseca: Se centra en evaluar al modelo directamente en la tarea general para la que fue entrenado (por ejemplo, la generación de texto). Su objetivo es medir qué tan bien se alinean las probabilidades predichas por el modelo con los datos reales.
- Perplejidad (Perplexity): Es la métrica estándar para este enfoque. Representa la inversa de la probabilidad de una secuencia (en un set de testeo), normalizada por la cantidad de palabras. Cuanto menor sea la perplejidad, mejor se considera que el modelo está prediciendo la muestra. Hay que tener precaución, ya que la perplejidad solo es comparable si los modelos fueron entrenados exactamente con el mismo corpus y bajo el mismo preprocesamiento. Además, lograr una perplejidad baja no siempre garantiza que el modelo vaya a rendir bien en una tarea del mundo real, por lo que siempre se debe complementar con evaluaciones extrínsecas.
2. Evaluación Extrínseca: Se enfoca en medir cómo rinde el modelo cuando se lo aplica a tareas concretas y específicas, como el reconocimiento de entidades nombradas (NER) o clasificación de secuencias.
- Métricas específicas: En este tipo de evaluación, las métricas varían según la tarea que se esté analizando. Se suelen utilizar métricas clásicas como la Accuracy (para ver la proporción general de predicciones correctas), el Área bajo la curva ROC (AUC-ROC) para medir la discriminación entre clases, así como la Precisión,

Sensibilidad (Recall) y el F1 Score. Para tareas de generación o traducción, se pueden usar métricas como BLEU y ROUGE.

3. Evaluación basada en Preferencias Humanas: Especialmente para modelos modernos que atraviesan fases de ajuste de instrucciones (Instruction Tuning), la evaluación se nutre directamente del feedback humano.
- Los evaluadores humanos rankean y comparan distintas respuestas del modelo basándose en qué tan informativas son, si intentan seguir las restricciones del usuario, y sus niveles de factualidad, alucinación o toxicidad.
- Con estas preferencias se suele entrenar un Reward Model (Modelo de

Recompensa), cuya única función es analizar un texto y devolver un puntaje (score) que indique qué tan bueno y alineado está el texto en relación con dichas preferencias humanas

## ¿Qué es perplexity?

La perplejidad (perplexity) es una métrica utilizada en la evaluación intrínseca para medir qué tan bien se alinean las probabilidades predichas por un modelo de lenguaje con los datos reales. Matemáticamente, se calcula como la inversa de la probabilidad de una secuencia de texto (en un conjunto de prueba), normalizada por la cantidad de palabras. De una forma más conceptual, representa el tamaño promedio del conjunto de palabras candidatas que el modelo está considerando en cada paso para hacer su siguiente predicción. Para ir al grano, estos son los puntos clave que debes anotar:

- Menor es mejor: Una perplejidad baja indica que el modelo está prediciendo la muestra con mayor precisión. Dada la enorme cantidad de parámetros en los modelos actuales, es una métrica robusta muy usada para monitorear el entrenamiento y evitar el sobreajuste.
- Regla estricta de comparación: Solo puedes comparar la perplejidad entre dos modelos distintos si ambos fueron evaluados utilizando el mismo conjunto de prueba, el mismo vocabulario y exactamente el mismo preprocesamiento. El tamaño del vocabulario y la forma en que se tratan las palabras desconocidas (marcadas como <UNK>) alteran drásticamente el cálculo matemático de esta métrica.
- No es una métrica definitiva: Lograr una baja perplejidad no siempre garantiza que el modelo vaya a tener un buen rendimiento al aplicarlo en tareas específicas reales.

Por ello, su uso siempre debe complementarse con evaluaciones extrínsecas

## ¿Qué relación hay entre word embeddings y redes neuronales?

Por que maridan tan bien? Los word embeddings y las redes neuronales "maridan" o se integran a la perfección por una razón fundamental: las redes neuronales requieren inputs (entradas) estrictamente numéricas para poder funcionar. El lenguaje natural es simbólico (letras y palabras), pero los embeddings resuelven este obstáculo transformando el texto en vectores numéricos densos y compactos que logran capturar el significado y el contexto (la semántica) de cada término. En Deep Learning, contar con estas excelentes representaciones matemáticas de los datos de entrada se traduce directamente en un mejor rendimiento del modelo. Esta relación es tan sinérgica que, en la práctica, se aprovecha de dos formas principales:

1. Como input ideal: Se pueden utilizar embeddings pre-entrenados (como Word2Vec) e inyectarlos directamente como las características (features) iniciales sobre las que trabajará y aprenderá la red neuronal.
2. Como parte del propio entrenamiento: Es posible incorporar el cálculo de los embeddings como una capa inicial dentro de la propia arquitectura de la red. Al hacer esto, mediante el algoritmo de backpropagation, la red neuronal ajustará y mejorará automáticamente los valores numéricos de los embeddings para optimizarlos de manera específica para la tarea final que está intentando resolver
## Diseñe un modelo de clasificación de texto (e.g., sentiment analysis) con una

red neuronal feed forward. Para diseñar un modelo de clasificación de texto (como sentiment analysis) utilizando una red neuronal feed forward, el proceso se estructura en los siguientes pasos, tomando como ejemplo la frase "Empanadas + Netflix, infalible!":

1. Embeddings como features (Capa de Entrada): El primer paso es transformar el texto simbólico en datos numéricos. Cada palabra de la frase (o token) se mapea a un vector denso utilizando embeddings pre-entrenados (como Word2Vec o GloVe).
2. Representación de la oración (Agregación): Dado que las oraciones tienen longitudes variables y una red feed forward estándar requiere una entrada de tamaño numérico fijo, necesitas colapsar todos los vectores de las palabras individuales en un único vector que represente a la oración completa. Esto se logra aplicando una operación sobre todos los vectores, calculando típicamente el promedio (element-wise mean) o el valor máximo (element-wise max) de sus componentes.
3. Capas Ocultas (Deep Neural Network): Una vez que tienes este vector único de tamaño fijo que resume el texto, lo inyectas directamente como entrada a las capas ocultas de tu red neuronal para que procese la información y extraiga los patrones.
4. Capa de Salida (Predicción): La última capa de la red procesa la información y devuelve un valor probabilístico. Por ejemplo, la red podría emitir un valor de , clasificando la oración como un sentimiento "Positivo"
## ¿Qué es BPE (Byte Pair Encoding)?

¿Cómo balancea el trade-off entre tamaño de vocabulario y largo de secuencia? El BPE (Byte Pair Encoding) es un método de tokenización por sub-palabras (subword tokenization) ampliamente utilizado en modelos modernos de lenguaje, como la familia GPT. Para ir al grano, su algoritmo funciona de la siguiente manera:

1. Comienza inicializando el vocabulario asumiendo que cada caracter individual es un token base.
2. Cuenta la frecuencia de todos los pares de tokens adyacentes que aparecen en el corpus de texto.
3. Toma el par que apareció con mayor frecuencia (por ejemplo, la letra "l" y la "o") y los "fusiona" creando un nuevo token único ("lo").
4. Repite este proceso iterativamente hasta alcanzar el número máximo de tokens (tamaño de vocabulario) deseado.
## ¿Cómo balancea el trade-off entre tamaño de vocabulario y largo de secuencia?

La tokenización clásica suele enfrentarse a dos extremos muy problemáticos:

- Si se tokeniza por palabras enteras: Las secuencias resultantes son cortas, pero el diccionario explota en tamaño tratando de abarcar todas las variaciones posibles, y el modelo falla ante palabras desconocidas (OOV).
- Si se tokeniza por caracteres individuales: El vocabulario es diminuto y no hay palabras desconocidas, pero las secuencias de texto se vuelven extremadamente largas computacionalmente, y un caracter por sí solo casi no captura información semántica.

BPE logra un equilibrio combinando lo mejor de ambos mundos: al fusionar iterativamente los caracteres más frecuentes, permite que las palabras comunes se transformen en un solo token (lo que mantiene la secuencia corta). Al mismo tiempo, si el modelo se cruza con una palabra rara o nueva, no se bloquea, sino que la divide en fragmentos o sub-palabras más pequeñas que ya conoce (por ejemplo, separando la palabra "lowest" en "low" y "est"). Esto garantiza que el tamaño del vocabulario se mantenga fijo y manejable, sin generar secuencias de texto ineficientemente largas

## ¿Qué son redes neuronales recurrentes?

Las redes neuronales recurrentes (RNR o RNN) son una arquitectura de Deep Learning diseñada específicamente para trabajar con datos secuenciales (como el texto o el audio), destacándose por su capacidad de "recordar" información de los pasos anteriores. Para ir al grano, su funcionamiento se basa en los siguientes mecanismos clave:

- Procesamiento Secuencial y Orden: Procesan los datos de entrada uno a uno. Esto les permite entender la posición de cada elemento, lo cual es vital en el lenguaje porque el orden de las palabras cambia completamente el significado de una oración.
- Estado Oculto (Memoria): Cada unidad o celda de la red posee un "estado oculto" () que actúa como una memoria interna que acumula el historial de la secuencia.
- Retroalimentación (Feedback): Tienen conexiones que se retroalimentan hacia sí mismas. En la práctica, para calcular el resultado de un paso específico, la capa oculta () se alimenta de la nueva entrada actual () y simultáneamente del estado oculto del paso temporal anterior (). Así, la información se propaga a través del tiempo, paso a paso, logrando una comprensión acumulativa.

Principales ventajas de esta arquitectura:

- Arrastre de contexto: A diferencia de los modelos Feed-Forward clásicos de contexto limitado, las RNR teóricamente permiten arrastrar información de todas las palabras anteriores de la secuencia, sin importar qué tan atrás estén.
- Flexibilidad de longitud: Al procesar paso por paso de forma cíclica, pueden ingerir secuencias de diferentes longitudes de forma dinámica, lo cual es ideal ya que las oraciones y documentos en el mundo real varían en tamaño.
- Eficiencia de memoria (Parámetros fijos): Los parámetros o pesos que utiliza la red son siempre los mismos en cada paso. Por lo tanto, el tamaño del modelo no aumenta ni depende del largo del texto de entrada
## Describa una célula recurrente, que elementos la componen?

Una célula recurrente es la unidad fundamental de una Red Neuronal Recurrente (RNR), encargada de procesar datos secuenciales paso a paso y de mantener y actualizar el "estado" o memoria de la red a medida que avanza. Para ir al grano, el flujo de una célula recurrente estándar se compone de los siguientes elementos matemáticos y estructurales:

- Entrada Actual (): Es el dato (por ejemplo, el embedding de una palabra) que la célula recibe en el paso de tiempo actual.
- Estado Oculto Previo (): Funciona como la memoria interna del modelo. Es la información que la red retroalimenta desde el paso inmediatamente anterior, lo que le permite a la célula tener en cuenta el contexto y la historia de lo ya procesado.
- Matrices de Pesos () y Sesgos (): Son los parámetros matemáticos fijos que la red aprende. El peso se multiplica por la entrada actual (), el peso se multiplica por el estado oculto previo (), y el peso se utiliza sobre el nuevo estado para calcular la salida. Un detalle crucial es que estos mismos pesos se reutilizan exactamente igual en cada paso de la secuencia.
- Funciones de Activación: Se utilizan operaciones matemáticas (como la función ) para sumar y comprimir la información de la entrada actual y la memoria anterior, generando así el nuevo estado oculto (). Finalmente, se puede usar otra función (como softmax) para emitir una predicción o salida final () en forma de probabilidad.

Adicionalmente, existen células recurrentes mucho más complejas diseñadas para solucionar problemas de memoria a muy largo plazo, como la célula LSTM (Long Short-Term Memory). Esta variante incorpora elementos extra:

- Cell State (): Una línea de "memoria" principal que atraviesa la célula.
- Compuertas (Gates): Mecanismos matemáticos que regulan el flujo de información en la memoria. Incluyen una Forget Gate (decide qué parte de la memoria vieja borrar), una Input Gate (decide qué información nueva actualizar en la célula) y una

Output Gate (decide qué parte del estado de la célula se enviará como salida)

## ¿Qué ventajas tienen las redes recurrentes con respecto al modelo ngrams en términos del largo de la secuencia?

La principal ventaja es que las redes neuronales recurrentes (RNR) pueden procesar secuencias de longitud variable y recordar información a largo plazo, superando la estricta limitación de "memoria corta" que tienen los modelos n-gram. El contraste entre ambos modelos en términos de longitud es el siguiente:

- El problema de los n-gramas: Están atrapados en una ventana fija de contexto (solo pueden mirar las últimas palabras). Esto les impide captar dependencias o relaciones a larga distancia en el texto. Si se intentara aumentar ese valor para darle más memoria al modelo, la cantidad de combinaciones explotaría y el modelo aumentaría su tamaño volviéndose inmanejable.
- La ventaja de las RNR (Arrastre de información): Las redes recurrentes procesan la secuencia paso a paso de forma acumulativa. Gracias a su mecanismo de memoria interna o estado oculto (), tienen la capacidad de "arrastrar" la información de las palabras anteriores, teóricamente tantas como se desee, sin importar qué tan atrás en la oración hayan aparecido.
- Independencia del tamaño: A diferencia de los modelos clásicos, en una red recurrente los pesos o parámetros matemáticos son siempre los mismos en cada paso de la secuencia. Por lo tanto, el tamaño del modelo no depende del largo de la secuencia de entrada, dándole una flexibilidad ideal para procesar oraciones cortas o documentos muy largos
## Describa las redes LSTMS

Las redes LSTM (Long Short-Term Memory) son una versión avanzada de las Redes Neuronales Recurrentes (RNR) clásicas, diseñadas específicamente para resolver los problemas de dependencia a largo plazo, lo que les otorga la capacidad de recordar información por períodos mucho más extensos en una secuencia de texto. Su innovación principal frente a una RNR simple radica en que incorporan una memoria adicional y un sistema de control interno llamado "compuertas":

- Cell State (Estado de la Celda): Además de mantener el estado oculto tradicional (), las LSTM introducen una nueva unidad llamada , que funciona como la "memoria" principal a largo alcance de la red. Es como una autopista central que atraviesa toda la célula, permitiendo que la información fluya fácilmente a través del tiempo sin diluirse.
- Forget Gate (Compuerta de Olvido): Es un mecanismo matemático que decide qué información vieja ya no es útil y debe descartarse (olvidarse) de la memoria.
- Input Gate (Compuerta de Entrada): Es la encargada de procesar el dato actual y decidir qué nueva información es relevante para actualizar y guardar en el estado de la celda.
- Output Gate (Compuerta de Salida): Finalmente, esta compuerta decide cuál será la salida de la red en ese paso específico, basándose puramente en la información filtrada que quedó almacenada en el estado de la celda.

Gracias a esta arquitectura de control de flujo, las LSTM son extremadamente eficientes y muy utilizadas en problemas complejos de predicción de secuencias

## ¿Qué ventajas tienen las LSTMS?

Las principales ventajas de las redes LSTM (Long Short-Term Memory) se resumen en los siguientes puntos:

- Evitan los problemas de dependencia a largo plazo: A diferencia de las redes neuronales recurrentes clásicas, las LSTM fueron diseñadas específicamente para recordar información por períodos mucho más largos a lo largo de una secuencia.
- Mitigan el desvanecimiento del gradiente (vanishing gradient): Al resolver el problema de pérdida de memoria, evitan este fenómeno matemático que degrada o desaparece la información de las primeras palabras a medida que la red procesa textos largos.
- Memoria estructural adicional: Logran estas ventajas porque, además de tener el estado oculto tradicional (), guardan la información de largo alcance en una unidad extra llamada Cell State (), la cual actúa como una memoria principal.
- Eficacia comprobada: Por su gran rendimiento, son muy utilizadas en problemas de predicción de secuencias, al punto de que se suelen usar por defecto como primera opción. En su variante bidireccional (Bi-LSTM), se convirtieron en la arquitectura más popular del Deep Learning para NLP gracias a su capacidad de capturar contexto en ambas direcciones
## ¿Cuál es la diferencia entre una red bidireccional y una unidireccional?

la diferencia fundamental radica en la forma en que procesan el contexto del texto y, en consecuencia, las tareas para las que resultan útiles:

- Redes Unidireccionales: Procesan la información en una sola dirección (generalmente de izquierda a derecha), por lo que un modelo unidireccional solo mira el pasado o el contexto previo. Matemáticamente, calculan la probabilidad de que aparezca una palabra basándose únicamente en la secuencia de palabras anteriores.

Gracias a esta naturaleza secuencial o autorregresiva, son las arquitecturas ideales para la generación de texto de forma directa, como es el caso de la familia GPT.

- Redes Bidireccionales: Procesan la secuencia de texto en ambas direcciones de manera simultánea: desde el principio hacia el final y desde el final hacia el principio. En el caso de las redes recurrentes (como las Bi-LSTM), esto se logra implementando dos redes separadas cuyos resultados u outputs luego se concatenan.

Su principal ventaja es que logran capturar tanto el contexto pasado como el contexto futuro de cada palabra.

## ¿Para qué se usa cada una?

Mientras que las redes unidireccionales dominan la generación, las bidireccionales son sumamente eficaces para tareas donde ya se tiene la secuencia de texto completa dada de antemano. Al tener una comprensión total del contexto que rodea a una palabra, son la arquitectura elegida para tareas analíticas como clasificación de textos, análisis de sentimientos o reconocimiento de entidades nombradas (NER). Como desventaja, al depender de mirar hacia adelante en la oración, no pueden utilizarse para generar texto nuevo de manera secuencial

## ¿Qué es ELMO y cuál es su arquitectura?

ELMo (Embeddings from Language Models) es un modelo de lenguaje bidireccional diseñado para generar embeddings contextuales. Su función principal es tomar embeddings estáticos como entrada y devolver representaciones dinámicas, donde el vector numérico de una palabra cambia dependiendo del contexto específico en el que se encuentre dentro de la oración. En cuanto a su arquitectura, se compone de los siguientes elementos clave:

- Entrada basada en caracteres (CNN): A diferencia de modelos más antiguos, el input inicial de ELMo utiliza representaciones basadas en caracteres que son procesadas a través de una red neuronal convolucional (CNN) para formar embeddings estáticos.
- LSTMs Bidireccionales: Su núcleo está formado por una red neuronal recurrente de múltiples capas que concatena redes LSTM entrenadas de manera independiente: una lee el texto de izquierda a derecha y la otra de derecha a izquierda. Específicamente, utiliza 2 capas con conexiones residuales entre ellas (con 4096 dimensiones ocultas y 512 dimensiones de salida).
- Jerarquía de aprendizaje: Las diferentes capas de la red se especializan en aprender características distintas del lenguaje. Las capas inferiores de las LSTM se centran en la sintaxis (la gramática y la estructura de las oraciones), mientras que las capas superiores capturan información más compleja orientada a la semántica (el significado y las relaciones profundas entre las palabras).

Finalmente, al ser un modelo que se pre-entrena con cantidades masivas de texto para predecir palabras en secuencias, sus representaciones ricas pueden ser luego ajustadas (fine- tuneadas) para resolver tareas específicas de Procesamiento de Lenguaje Natural

## ¿Cómo diseñarías un problema de clasificación con ELMO?

Diseñar un problema de clasificación utilizando ELMo implica aprovechar sus representaciones contextuales pre-entrenadas y adaptar la arquitectura final mediante un proceso de fine-tuning. Basándonos en los esquemas de las fuentes, los pasos son los siguientes:

1. Generación de Embeddings (Capa base): Pasas tu texto de entrada por el modelo ELMo para generar los embeddings contextuales de cada palabra de la secuencia.
2. Adaptación de la representación (Agregación): Como ELMo devuelve vectores individuales por cada palabra, necesitas procesarlos para obtener un vector único que represente toda la información de la oración. Esto se logra aplicando una capa de Global Max Pooling directamente sobre los embeddings de ELMo, o bien pasándolos por una unidad recurrente adicional (como una red GRU) y tomando su último estado oculto (last hidden state).
3. Capa de clasificación (Head): Ese vector representativo se inyecta en una capa densa (Dense layer) y luego pasa a un clasificador Softmax, el cual se encargará de emitir la probabilidad y la etiqueta predicha final (Predicted label).
4. Proceso de Fine-Tuning (Ajuste Fino): Con la arquitectura ensamblada, procedes a entrenar el modelo con tus datos específicos de clasificación. Se utiliza la métrica de Cross-entropy para comparar la salida del Softmax con las etiquetas reales de tu tarea, optimizando los hiperparámetros para que el modelo aprenda las particularidades de tu problema
## ¿Qué es pre training?

El pre-entrenamiento (o pre-training) es la fase inicial y fundamental en el desarrollo de un modelo de lenguaje, en la que se lo entrena sobre un corpus de texto masivo para resolver una tarea general. Sus características y ventajas principales son:

- Aprendizaje autosupervisado: Al utilizar enormes cantidades de datos sin etiquetar por humanos, el proceso define "tareas pretexto" para que el modelo genere sus propios ejemplos y aprenda por sí solo. Las tareas más comunes de pre-entrenamiento incluyen predecir una palabra que ha sido enmascarada u oculta en una oración (como hace BERT), predecir secuencialmente cuál es la siguiente palabra (como hace GPT), o predecir si una oración lógicamente le sigue a otra.
- Adquisición de conocimiento: Gracias a esta fase, el modelo logra aprender representaciones sumamente ricas del lenguaje. Comprende desde la sintaxis y las estructuras gramaticales, hasta los matices semánticos y un amplio conocimiento general del mundo a través de los textos que procesa.
- Aceleración y Transferencia de Aprendizaje (Transfer Learning): Una vez que el modelo está pre-entrenado, cuenta con una base tan sólida de patrones lingüísticos que requiere de muchísimo menos tiempo y una dependencia drásticamente menor de datos etiquetados para ser ajustado (fine-tuneado) a una tarea específica. En modelos de muy gran escala, la necesidad de usar datos extra para una tarea puede incluso reducirse a cero
## ¿Cuál es el paradigma de pre-training y fine-tuning?

El paradigma de pre-training (pre-entrenamiento) y fine-tuning (ajuste fino) es la estrategia central del Transfer Learning (Aprendizaje Transferido) en Inteligencia Artificial, estructurada en dos grandes etapas para crear modelos altamente eficientes y precisos. Funciona de la siguiente manera:

- 1. Pre-training (Cimientos generales): En esta primera fase, el modelo se entrena sobre una cantidad masiva de datos de texto general y sin etiquetar (como artículos de Wikipedia o páginas web). Para aprender, resuelve tareas autosupervisadas genéricas, como adivinar qué palabra sigue en una oración o deducir una palabra que ha sido ocultada. El objetivo no es resolver un problema específico, sino que el modelo adquiera una comprensión profunda y rica del lenguaje, aprendiendo estructuras sintácticas, semántica y conocimiento general del mundo.
- 2. Fine-tuning (Especialización): En la segunda fase, ese modelo "educado" se adapta para resolver un problema concreto y particular, como clasificar textos, analizar sentimientos o responder preguntas. Para lograrlo, se lo entrena con un conjunto de datos especializados y mucho más pequeño, guiándolo para que aprenda los detalles y particularidades de esa tarea objetivo.

La clave de su éxito: Si intentáramos entrenar un modelo desde cero para una tarea compleja, empezaríamos con una inicialización aleatoria de sus parámetros matemáticos, lo cual es ineficiente y requiere enormes cantidades de datos etiquetados. Con este paradigma, el modelo comienza la fase de fine-tuning con pesos que ya están muy cerca del punto óptimo. Al aprovechar todo el conocimiento previo, se acelera drásticamente el tiempo de entrenamiento y se reduce (a veces casi a cero en modelos inmensos) la necesidad de recolectar datos manuales costosos para la tarea final

## ¿Qué diferencia hay entre embeddings estáticos y contextuales?

La diferencia principal radica en si el vector numérico que representa a una palabra cambia o se mantiene igual dependiendo de la oración en la que se encuentra:

- Embeddings Estáticos: Cada palabra del vocabulario tiene asignado un único vector matemático fijo e independiente del contexto. o El problema de la polisemia: Al tener una sola representación por palabra, no pueden distinguir términos con múltiples significados. Por ejemplo, le asignarán exactamente el mismo vector a la palabra "banco" en la frase "banco financiero" que en la frase "banco de plaza".

o Características: Son menos costosos computacionalmente y se entrenan basándose en estadísticas de co-ocurrencia global, sin analizar dinámicamente cada oración nueva. Modelos clásicos como Word2Vec, GloVe y FastText generan este tipo de embeddings.

- Embeddings Contextuales: El vector de una palabra es dinámico y se calcula en el momento dependiendo específicamente de las demás palabras que la rodean en la oración. o Solución a la polisemia: Como logran comprender la oración completa, generan representaciones distintas según el uso. El vector para "banco" será muy diferente en la frase "Saqué dinero del banco" que en "Me siento en el banco".

o Características: Requieren arquitecturas de redes neuronales más complejas que procesan secuencias enteras. Son la base del NLP moderno y ejemplos de modelos que los generan incluyen a ELMo, BERT y GPT

## ¿Qué ventaja tiene un embedding contextual frente a un estático?

La ventaja fundamental de un embedding contextual frente a uno estático es su capacidad para resolver la polisemia y la ambigüedad del lenguaje. Mientras que un modelo estático (como Word2Vec o GloVe) le asigna un único vector fijo a cada palabra de forma aislada, un modelo contextual (como ELMo o BERT) analiza la oración completa y genera una representación dinámica que depende específicamente de las palabras que la rodean. La ventaja práctica se hace evidente con palabras que tienen múltiples significados. Por ejemplo, ante la palabra "banco", un modelo estático utilizará el mismo vector matemático sin importar de qué se esté hablando. En cambio, el embedding contextual logrará comprender el entorno de la oración y le asignará un vector muy distinto a la palabra "banco" en la frase "Saqué dinero del banco" en comparación con la frase "Me siento en el banco". Esto es posible porque, en su forma de entrenamiento, los modelos contextuales utilizan arquitecturas neuronales avanzadas que logran comprender oraciones completas, superando a los modelos estáticos que se limitan a analizar métricas de co-ocurrencia global sin un contexto dinámico real

## ¿Cómo se generan embeddings contextuales?

Los embeddings contextuales se generan procesando una oración o secuencia completa a través de arquitecturas neuronales avanzadas (como las redes recurrentes bidireccionales en ELMo o los Transformers en BERT o GPT) que son capaces de comprender las oraciones completas y el contexto que rodea a cada término. El proceso general funciona de la siguiente manera:

1. Entrada inicial: El modelo no procesa las palabras de forma aislada. Recibe la secuencia entera y toma las representaciones base de esas palabras (que inicialmente pueden ser embeddings estáticos o representaciones basadas en caracteres, como en el caso de ELMo).
2. Análisis del entorno: A medida que la información avanza por las múltiples capas de la red neuronal, el modelo analiza y combina matemáticamente la información de las palabras vecinas. Por ejemplo, en modelos bidireccionales, la red lee simultáneamente el texto de izquierda a derecha y de derecha a izquierda para capturar el panorama completo.
3. Generación dinámica (Salida): Finalmente, las últimas capas de la red devuelven un nuevo vector numérico para cada palabra. Como este cálculo final incorpora las interacciones con el resto de los términos de esa oración específica, el output es una representación dinámica; es decir, el vector de una misma palabra cambiará dependiendo del contexto exacto en el que fue utilizada Explica el mecanismo de atención, como funciona, intuición, etc El mecanismo de atención se basa en una intuición muy humana: al procesar información o leer una oración, no le damos la misma importancia a todas las palabras por igual. Funciona como un "selector de inputs", permitiendo que el modelo se enfoque y preste mayor atención a las partes específicas del texto que son más útiles para tomar una decisión en un momento dado. En las fuentes se ilustra este concepto con un esquema muy claro de análisis de sentimiento para la oración "El hotel está buenísimo". Para clasificar esta frase como un comentario "Positivo", el modelo no analiza todo por igual; el mecanismo de atención le asigna un peso o puntaje altísimo de (95%) a la palabra clave "buenísimo", mientras que a palabras de relleno como "El" , "hotel" o "está" les asigna pesos casi nulos (entre y ). Su funcionamiento matemático se divide en tres pasos:
1. Ponderación de Entradas: El modelo analiza la secuencia y le asigna un "peso" o puntaje a cada parte de la entrada basándose en qué tan relevante es.
2. Combinación Contextual: Se crea un "vector de contexto", que es simplemente una suma matemática donde las entradas se mezclan según el peso que recibieron. Lo importante brilla, lo irrelevante se opaca.
3. Generación de Salida: Ese nuevo resumen enriquecido y focalizado se utiliza finalmente para emitir la predicción de la red.
## ¿Qué es el mecanismo de atención?

¿Cuáles son sus ventajas? Como definimos en la respuesta anterior, es un componente de red neuronal que pondera dinámicamente las partes de un texto según su importancia. Al incorporarse (y luego reemplazar) a arquitecturas previas como las Redes Recurrentes, aportó ventajas revolucionarias que sentaron las bases del procesamiento de lenguaje moderno:

- Captura de Relaciones a Largo Plazo: Los modelos antiguos procesaban las secuencias paso a paso y terminaban "olvidando" lo que leyeron al principio. El mecanismo de atención soluciona esto permitiendo que cualquier palabra "vea" y se conecte directamente con todas las demás palabras de la secuencia de forma simultánea, sin importar qué tan lejos estén físicamente en la oración.
- Mejora del Rendimiento: Al aliviar los cuellos de botella de información y evitar que los gradientes matemáticos se desvanezcan, permite entrenar modelos mucho más robustos y precisos, mejorando enormemente tareas complejas como la traducción automática.
- Interpretabilidad (Apertura de la "caja negra"): En el Deep Learning suele ser difícil saber cómo piensa la máquina. Sin embargo, los pesos numéricos que genera el mecanismo de atención nos brindan una radiografía perfecta para auditar el modelo: podemos visualizar exactamente a qué fragmentos de texto le prestó atención la red para llegar a su conclusión (como en el diagrama donde vimos que focalizó el

95% de su atención en "buenísimo")

## ¿Cómo relacionas el mecanismo de atención con la arquitectura transformers?

la relación es absoluta y fundacional: la arquitectura Transformer está construida puramente sobre el mecanismo de atención, eliminando por completo la necesidad de utilizar redes neuronales recurrentes (RNR) para procesar el lenguaje. De hecho, el paper que introdujo los Transformers en 2017 se titula, justamente, "Attention is all you need" (La atención es todo lo que necesitas). Esta estrecha integración se manifiesta en la arquitectura a través de los siguientes puntos clave:

- El motor de Auto-Atención (Self-Attention): Mientras que el mecanismo de atención original se usaba como un "puente" entre dos redes distintas, los

Transformers usan auto-atención. Esto significa que cada palabra (token) de la oración calcula su propia representación observando y ponderando a todas las demás palabras de su misma secuencia al mismo tiempo. Para lograrlo, el modelo deriva tres vectores a partir de cada palabra: Queries (Consultas), Keys (Claves) y Values (Valores), los cuales se operan matemáticamente para calcular el puntaje (score) de atención.

- Procesamiento Paralelo: Las redes recurrentes procesan el texto paso a paso de forma lineal, lo que genera cuellos de botella. Al basarse 100% en la atención, los

Transformers ingieren toda la secuencia en paralelo. El mecanismo de atención permite que cualquier palabra mire y se relacione directamente con cualquier otra en un solo paso matemático, sin importar qué tan lejos estén en la oración, capturando perfectamente las dependencias a largo plazo y siendo drásticamente más rápidos de entrenar.

- Atención Multi-Cabeza (Multiheaded Attention): Los Transformers no calculan la atención una sola vez, sino que ejecutan múltiples mecanismos de atención en paralelo dentro de cada capa. Estas múltiples "cabezas" le permiten al modelo centrarse en diferentes partes o características del texto simultáneamente, y al apilar varios de estos bloques, la red logra extraer información y relaciones gramaticales y semánticas profundas.
- Atención Cruzada (Encoder-Decoder Attention): En las arquitecturas Transformer diseñadas para traducción o resúmenes, que cuentan tanto con un codificador como con un decodificador, se incluye una capa de atención adicional. Esta se encarga exclusivamente de que el decodificador le "preste atención" a la representación compacta del texto original que generó el codificador, guiando así la generación de la nueva secuencia
## ¿Qué ventajas tiene transformers frente a las redes neuronales recurrentes?

Las arquitecturas Transformer presentan ventajas fundamentales sobre las Redes Neuronales Recurrentes (RNN) que solucionaron los cuellos de botella clásicos del procesamiento de secuencias:

- Procesamiento en paralelo: Mientras que las RNN están forzadas a procesar los datos de manera secuencial y lineal (paso a paso), los Transformers procesan todos los elementos de la secuencia de texto simultáneamente. Esta característica los hace significativamente más rápidos de entrenar y altamente escalables.
- Captura de dependencias a largo plazo: Las redes recurrentes sufren de pérdida de información cuando intentan recordar palabras lejanas debido al problema del desvanecimiento del gradiente. Los Transformers resuelven este obstáculo gracias a su mecanismo de atención, el cual les permite modelar y retener dependencias lejanas con gran eficacia.
- Interacción directa e inmediata: En una RNN, las palabras no interactúan directamente; a medida que la distancia física entre dos tokens aumenta, su relación matemática se desvanece. En cambio, el mecanismo de atención de los Transformers logra que cada palabra "vea", analice y pondere a todas las demás palabras de la oración en un solo paso. Esto genera una representación y una comprensión del contexto muchísimo más rica.
- Arquitectura flexible: Su diseño modular permite utilizar distintos tipos de atención y diferentes tipos de capas, haciéndolos mucho más adaptables para resolver diversas tareas o aplicarse en distintos dominios
## ¿Podes hacer un overview de la arquitectura transformers?

la arquitectura Transformer (introducida en el famoso paper "Attention is all you need" de 2017) revolucionó el procesamiento del lenguaje al eliminar por completo el uso de redes recurrentes y basar su diseño puramente en mecanismos de atención, logrando procesar secuencias enteras en paralelo. Basándonos en las fuentes, un overview de su arquitectura original se divide en los siguientes componentes fundamentales:

1. Entrada y Positional Encoding (Codificación Posicional) Como los Transformers ingieren toda la secuencia de texto al mismo tiempo en lugar de leer palabra por palabra, carecen de una noción natural del orden. Para solucionar esta "ausencia de orden", se le suma un vector especial (Positional Encoding) a cada embedding original de la palabra justo antes de entrar al modelo, indicándole matemáticamente su posición exacta en la oración.
2. El Motor: Self-Attention y Multi-Head Attention El núcleo de la arquitectura es el mecanismo de Self-Attention (Auto-Atención). A través de transformaciones lineales que generan vectores llamados Queries (Q), Keys (K) y Values (V), cada palabra calcula un puntaje (score) de atención ponderando a todas las demás palabras de la oración para formar su representación. En lugar de hacerlo una sola vez, la arquitectura utiliza múltiples cabezas en paralelo (Multi-Head Attention), lo que permite que la red se concentre en distintos aspectos y matices semánticos al mismo tiempo, concatenando finalmente todos los resultados.
3. El Codificador (Encoder) Su objetivo es analizar la entrada y generar una representación matemática riquísima del contexto. Está compuesto por una pila de capas idénticas que contienen:
- Una capa de Multi-Head Attention.
- Una red neuronal tradicional densa (Feed Forward).
- Add & Norm: Alrededor de las dos capas anteriores, se utilizan conexiones residuales (atajos matemáticos que suman la entrada directamente a la salida para evitar que se pierda información profunda) seguidas de una Normalización de Capa (Layer Normalization) que estabiliza y acelera enormemente el entrenamiento.
4. El Decodificador (Decoder) Su objetivo es generar la salida paso a paso (de forma autorregresiva). Su diseño es muy similar al del Encoder, pero con adiciones críticas:
- Atención Enmascarada (Masked Multi-Head Attention): En la generación de texto, el modelo no puede hacer trampa mirando el futuro. Por lo tanto, se aplica una "máscara causal" triangular que oculta los tokens que aún no se generaron (reemplazándolos por un valor de ), obligando al modelo a guiarse solo por el pasado.
- Atención Cruzada (Encoder-Decoder Attention): Incluye una sub-capa intermedia especial. Aquí, las consultas (Queries) vienen de la capa anterior del Decoder, pero las claves y valores (Keys y Values) provienen de la salida final del Encoder. Esto es lo que permite que el generador "le preste atención" al texto original que procesó el

Encoder (fundamental en tareas como la traducción). A partir de esta estructura completa de Encoder-Decoder, el ecosistema de IA luego se ramificó: se crearon arquitecturas de solo-Encoder (como BERT, excelentes para clasificar) y arquitecturas de solo-Decoder (como GPT, líderes en la generación de texto)

## ¿Cómo soluciona el Transformers la falta de sentido posicional (opuesto a las RNN)?

Los Transformers solucionan su "ausencia de orden" utilizando un mecanismo llamado Positional Embeddings (Codificaciones Posicionales). El problema original surge porque, a diferencia de las Redes Neuronales Recurrentes (RNN) que procesan el texto paso a paso e incorporan la posición de forma natural, los Transformers procesan todas las palabras de la secuencia de manera simultánea o en paralelo. Al ingerir todos los datos de golpe, el modelo por defecto no tiene forma de saber qué palabra va primero o última. Esto es un problema crítico, ya que en el lenguaje el orden de las palabras altera drásticamente el significado de la oración. Para resolverlo, la arquitectura le suma un vector matemático especial al embedding original de cada palabra justo antes de que pase al modelo. Este vector actúa como una "etiqueta" que le provee a la red la información exacta sobre la posición de ese token en la secuencia. A medida que las arquitecturas avanzaron, la forma de calcular esta posición evolucionó en tres métodos principales:

1.
2.
3. Sinusoidal (Transformer original): Utiliza una fórmula matemática fija basada en

senos y cosenos que genera un vector único por cada posición. Su principal ventaja es que no requiere entrenamiento extra y puede calcular posiciones para secuencias de texto de cualquier longitud. Learned (Entrenados): Modelos como BERT o GPT-2 tratan a la posición como un embedding más que la red debe aprender durante el entrenamiento. Aunque el modelo aprende exactamente qué posiciones importan, tiene la desventaja de estar estrictamente limitado a la longitud máxima de texto con la que fue entrenado (no generaliza bien a textos más largos). RoPE (Rotary Position Embeddings): Es el estándar en los grandes modelos de lenguaje actuales como GPT-4, LLaMA y Mistral. En lugar de sumar un vector, este método rota matemáticamente las dimensiones del embedding original en un ángulo que es proporcional a su posición. Es sumamente eficaz porque logra que al calcular la atención, a la red le importe únicamente la distancia relativa entre dos palabras (qué tan lejos están una de la otra) y no su posición absoluta, permitiéndole generalizar a secuencias de texto inmensas sin utilizar parámetros adicionales

## ¿Que es Self attention?

El Self-Attention (auto-atención) es un mecanismo que permite a un modelo centrarse simultáneamente en diferentes partes de una misma secuencia de entrada para comprender el contexto. A diferencia del mecanismo de atención clásico donde las entradas provienen de lugares distintos, en la auto-atención los tres vectores matemáticos utilizados provienen exactamente de la misma fuente (los embeddings de entrada de la oración). Estos vectores son: Queries (Q), Keys (K) y Values (V). Funciona a través de los siguientes pasos:

1.
2.
3. Puntuación: El modelo calcula un score o puntuación matemática cruzando el Query

de una palabra con las Keys del resto de las palabras. Ponderación: Esos scores pasan por una función softmax para convertirse en pesos de atención, determinando cuánta importancia debe prestarle a cada término de la secuencia. Representación final: La nueva representación de la palabra se forma calculando una suma ponderada de los Values (V) de toda la secuencia, utilizando los pesos de atención obtenidos en el paso anterior

## ¿Qué son los positional embeddings?

Los positional embeddings (o codificaciones posicionales) son vectores matemáticos diseñados para proporcionar a los modelos Transformer información sobre la posición u orden de cada palabra dentro de una secuencia. Surgen para solucionar la "ausencia de orden" de esta arquitectura: al procesar todos los datos de entrada en paralelo (a diferencia de las RNNs), el modelo por defecto no tiene forma de saber qué palabra va primero, algo esencial para entender el significado del texto. Para resolverlo, estas codificaciones se suman directamente a los embeddings originales de las palabras justo antes de pasar por el modelo. La forma de implementarlos ha evolucionado en tres métodos principales:

- Sinusoidales: Utilizan una fórmula matemática fija basada en senos y cosenos para generar un vector único por posición, permitiendo escalar a secuencias de cualquier longitud.
- Entrenados (Learned): El modelo trata a la posición como un embedding más que debe aprender durante el entrenamiento (usado en BERT o GPT-2).
- RoPE (Rotary Position Embeddings): Es el estándar actual (usado en LLaMA, GPT- 4, etc.). En lugar de sumar un vector, rota matemáticamente las dimensiones del embedding original en un ángulo proporcional a su posición. Esto hace que el modelo se enfoque puramente en la distancia relativa entre las palabras, generalizando muy bien a secuencias largas
## ¿Qué es el encoder?

El encoder (o codificador) es el componente encargado de procesar el texto de entrada para crear una representación matemática compacta y rica en contexto. Esta representación es la que luego utiliza el componente "decoder" (decodificador) para poder generar el texto de salida. Al observar su estructura dentro de la arquitectura Transformer, cada bloque del encoder está formado por dos subcapas principales:

- Atención Multi-Head: El mecanismo que permite que la red pondere y relacione las palabras de la secuencia.
- Red Feed-Forward (FFN): Una red neuronal tradicional densa.

Adicionalmente, después de cada una de estas subcapas se aplica una conexión residual seguida de una normalización de capa. La clave del encoder radica en su estado oculto (hidden state). A medida que los datos de entrada pasan por las múltiples capas del encoder, este estado oculto se actualiza y refina continuamente a través de las operaciones de atención y la red FFN, acumulando la información procesada de forma bidireccional para aprovechar al máximo el contexto de la oración. Históricamente, en arquitecturas más antiguas como el Seq2Seq tradicional, el encoder sufría de un "cuello de botella" porque estaba forzado a comprimir toda la entrada en un único vector fijo final (su último estado oculto), lo que provocaba que la información de las primeras palabras se perdiera o degradara al procesar textos muy largos

## Describí el input the BERT.

El input de BERT se construye matemáticamente a partir de la suma de tres tipos de embeddings distintos para cada token:

1. Token Embeddings: Es la representación de la palabra o subpalabra. Para generarlos, BERT utiliza un tokenizador llamado WordPiece.
2. Segment Embeddings: Son vectores que le indican al modelo a qué segmento u oración específica pertenece cada token. Resultan fundamentales cuando la entrada está compuesta por pares de oraciones (como el formato pregunta/respuesta).
3. Position Embeddings: Son vectores entrenables (learned) que se suman para indicarle a la red la posición exacta de cada token dentro de la secuencia (generalmente limitados a una longitud máxima de 512 posiciones). A nivel de formato, la secuencia de entrada siempre requiere la incorporación de dos símbolos especiales:
- [CLS]: Se coloca obligatoriamente al inicio de cada entrada. El modelo utilizará la representación final de este token específico para realizar predicciones en tareas de clasificación.
- [SEP]: Funciona como un token separador (por ejemplo, para dividir claramente dónde termina la primera oración y dónde empieza la segunda)
## Diseña una tarea de clasificación con BERT.

Diseñar una tarea de clasificación con BERT requiere adaptar su modelo pre-entrenado a través del proceso de fine-tuning. La arquitectura se estructura de la siguiente manera:

1. Formato de Entrada: Todo texto que ingresa al modelo debe comenzar obligatoriamente con el token especial [CLS]. La representación inicial de cada palabra se construye sumando sus token embeddings, segment embeddings y position embeddings.
2. Procesamiento Base: Esta secuencia atraviesa todas las capas del Transformer encoder bidireccional de BERT para generar representaciones ricas en contexto.
3. Extracción del vector representativo: A diferencia de otros modelos que promedian la salida de todas las palabras, para clasificar una oración completa en BERT se aísla únicamente el vector de salida final que corresponde al token [CLS]. El modelo es entrenado para que la representación de este token específico condense toda la información de la secuencia.
4. Capa de Clasificación (Task Head): El vector del token [CLS] se inyecta directamente en una nueva capa densa o lineal que se coloca en la parte superior del modelo.
5. Entrenamiento (Fine-Tuning): Durante el entrenamiento con el conjunto de datos específico de la tarea (por ejemplo, reseñas etiquetadas como positivas o negativas), los pesos de esa nueva capa de clasificación y todos los parámetros internos de BERT se ajustan simultáneamente. El modelo optimiza sus predicciones calculando el error mediante una métrica como Cross-entropy sobre las etiquetas reales
## ¿Cómo harías para anonimizar entidades (personas, direcciones, etc) de fallos judiciales?

Para anonimizar entidades como personas o direcciones en fallos judiciales, el enfoque ideal es abordarlo como una tarea de Reconocimiento de Entidades Nombradas (NER, por sus siglas en inglés). El proceso paso a paso se estructuraría de la siguiente manera:

1.
2.
3.
4. Elección del modelo: Utilizarías una arquitectura bidireccional que genere embeddings

contextuales (como BERT o ELMo), ya que son los indicados para comprender el contexto de la oración y clasificar tokens individuales. Entrenamiento (Fine-tuning): Ajustarías este modelo pre-entrenado utilizando un corpus de textos legales donde las palabras ya estén previamente etiquetadas en categorías específicas, como PER (persona), LOC (localización o dirección) u ORG (organización). Detección: Al pasar el texto de un nuevo fallo judicial por el modelo, la red neuronal analizará cada palabra y emitirá una predicción, etiquetando los fragmentos de texto que correspondan a entidades sensibles. Anonimización: Una vez que el modelo te devuelve el texto con sus etiquetas, simplemente aplicas un script que busque todas las palabras clasificadas como "PER" o "LOC" y las reemplace por una máscara genérica (por ejemplo, cambiando "Juan Pérez" por "[PERSONA]"), logrando así anonimizar el documento de forma automática.

## ¿Qué diferencia hay entre un modelo bidireccional y uno unidireccional?

## Como se manifiesta la diferencia en términos de attention?

La diferencia fundamental reside en la cantidad de contexto que pueden procesar. Un modelo bidireccional analiza la secuencia de texto en ambas direcciones simultáneamente, capturando tanto el contexto pasado como el futuro de cada palabra. Esto lo hace ideal para tareas analíticas como clasificación, pero incapaz de generar texto nuevo secuencialmente. Por el contrario, un modelo unidireccional procesa la información de forma secuencial (generalmente de izquierda a derecha), observando únicamente el pasado, lo que lo convierte en la arquitectura estándar para la generación de texto. En la arquitectura Transformer, esta diferencia se manifiesta en el mecanismo de atención a través del enmascaramiento (masking):

- En modelos bidireccionales (como los Encoders): Se utiliza el mecanismo de Self- Attention regular, donde la matriz de atención permite que cada token calcule su representación observando libremente a todos los demás tokens de la secuencia de entrada al mismo tiempo.
- En modelos unidireccionales (como los Decoders): Se implementa una variante llamada Causal Masking (o Masked Self-Attention). Para forzar al modelo a mirar solo hacia el pasado, se aplica una "máscara" que oculta los tokens futuros reemplazando sus valores matemáticos por justo antes de aplicar la función Softmax.

Esto anula su peso, asegurando que cada palabra preste atención estrictamente a las palabras anteriores y no haga trampa "viendo el futuro" durante la generación

## ¿Qué son las conexiones residuales y qué rol juegan en la arquitectura transformers?

Las conexiones residuales son "atajos" en la arquitectura de la red neuronal que saltan una o más capas intermedias sumando la entrada original directamente a la salida generada por esa capa. Dentro de la arquitectura Transformer, estas conexiones se ubican inmediatamente después de cada subcapa principal (es decir, después del mecanismo de Multi-Head Attention y de la red Feed-Forward) y van seguidas de un paso de normalización (Layer Normalization). Juegan un rol crítico para el éxito del modelo y aportan las siguientes ventajas prácticas:

- Solución al desvanecimiento del gradiente: Al proporcionar un camino directo hacia las primeras capas, permiten que la información del error fluya hacia atrás durante el entrenamiento sin diluirse. Esto estabiliza la red y acelera drásticamente su convergencia.
- Preservación de la información: Aseguran que la información vital y fundamental de cada token no se pierda a medida que los datos atraviesan las múltiples capas de la red.
- Refinamiento incremental: Permiten que el modelo realice ajustes aditivos (sumas) a los estados ocultos en lugar de transformarlos por completo. Por ejemplo, al procesar la frase "hombre araña", la conexión residual ayuda a que el modelo conserve la esencia semántica original de la palabra "hombre" mientras simplemente le añade o refina el nuevo contexto aportado por la palabra "araña"
## ¿Qué es un decoder?

El decoder (o decodificador) es el componente de una arquitectura neuronal diseñado para generar texto de salida paso a paso, de forma autorregresiva. En arquitecturas compuestas (como el Encoder-Decoder), el decoder funciona como un modelo de lenguaje condicionado: recibe la representación matemática compacta que procesó el encoder y la utiliza para producir la secuencia final, siendo ideal para tareas generativas como la traducción automática o el resumen de textos. Dentro de la arquitectura Transformer, cada capa del decoder tiene características únicas:

- Causal Masking (Atención Enmascarada): Implementa una máscara que oculta las palabras que aún no se generaron (reemplazándolas matemáticamente por ). Esto fuerza al modelo a prestar atención únicamente a los tokens anteriores, impidiendo que "vea el futuro" y haga trampa mientras genera el texto.
- Atención Cruzada (Encoder-Decoder Attention): Posee una subcapa de atención adicional que se centra exclusivamente en conectar al decoder con la salida del encoder, permitiéndole extraer la información del texto original.

Además, existen arquitecturas modernas basadas exclusivamente en este componente (modelos Decoder-only, como la familia GPT). Estos modelos prescinden del encoder y se especializan netamente en la generación de texto, entrenándose bajo el objetivo continuo de predecir cuál es la próxima palabra basándose en el historial de tokens previos

## Tengo un problema de clasificación, tengo los datos, ¿cómo lo resolverías?

Para resolver un problema de clasificación de texto teniendo ya los datos etiquetados, el enfoque más efectivo es utilizar el paradigma de Transfer Learning mediante el ajuste fino (fine-tuning) de un modelo de lenguaje pre-entrenado. El proceso paso a paso para resolverlo sería el siguiente:

1. Elección del modelo base: Elegiría una arquitectura Transformer bidireccional, como BERT, ya que al procesar el contexto en ambas direcciones es ideal para extraer representaciones profundas de una secuencia. Al utilizar un modelo pre-entrenado sobre datos generales, sus pesos matemáticos ya parten desde un punto mucho más cercano al óptimo.
2. Formato de entrada: Adaptaría tus datos al formato estricto que requiere el modelo. Para BERT, esto implica tokenizar el texto y agregar obligatoriamente el token especial [CLS] al inicio de cada oración o documento.
3. Procesamiento y extracción: Al pasar tus textos por el modelo, este generará embeddings contextuales para cada palabra. Para la clasificación, se debe aislar únicamente el vector de salida final que corresponde al token [CLS], ya que este condensa toda la información de la secuencia completa.
4. Capa especializada (Task Head): Inyectaría ese vector representativo del token [CLS] directamente en una nueva capa neuronal densa configurada para la cantidad de categorías específicas que tenga tu problema de clasificación.
5. Entrenamiento (Fine-Tuning): Entrenaría esta arquitectura con tus datos especializados. Durante esta fase, tanto los pesos de la nueva capa final como los parámetros internos del modelo base se ajustan simultáneamente para la tarea objetivo, calculando el error de las predicciones frente a tus etiquetas reales mediante una métrica como Cross-entropy.
6. Evaluación: Finalmente, evaluaría la calidad del modelo utilizando métricas de selección como el Accuracy (para conocer la proporción total de clasificaciones correctas), la Precisión (para saber cuántas clasificaciones positivas fueron correctas), o el área bajo la curva ROC (AUC-ROC) (para medir la habilidad del modelo para discriminar correctamente entre las distintas clases)
## Tengo un problema de NER, tengo los datos, ¿cómo lo resolverías?

Para resolver un problema de Reconocimiento de Entidades Nombradas (NER) teniendo los datos etiquetados, aplicaría el paradigma de Transfer Learning mediante el ajuste fino (fine- tuning) de un modelo pre-entrenado, enfocándolo específicamente como una tarea de etiquetado de secuencias (sequence labeling). El proceso paso a paso sería el siguiente:

1. Elección del modelo base: Utilizaría una arquitectura neuronal bidireccional, como BERT, ELMo o una red Bi-LSTM. Al procesar el texto en ambas direcciones simultáneamente, el modelo logra capturar el contexto completo que rodea a cada término, lo cual es fundamental para identificar entidades correctamente.
2. Generación de representaciones: Al ingresar tus textos, el modelo base pre- entrenado generará representaciones dinámicas (embeddings contextuales) para cada palabra.
3. Capa de clasificación por token: A diferencia de un problema de clasificación de texto tradicional (donde se predice una única etiqueta para toda la oración), en NER se debe evaluar cada elemento individualmente. La representación matemática de cada token se conectará a una capa clasificadora que predecirá a qué categoría específica pertenece (por ejemplo, etiquetando palabras como PER para persona, LOC para localización u ORG para organización, o indicando que no son ninguna entidad).
4. Entrenamiento (Fine-Tuning): Entrenaría esta arquitectura con tus datos especializados. Durante esta fase, el modelo optimizará sus predicciones calculando el error mediante la métrica de Cross-entropy sobre las etiquetas reales. Gracias al conocimiento previo del modelo base, este ajuste fino requerirá muchos menos datos manuales para lograr un buen rendimiento.
5. Evaluación: Finalmente, para medir de forma objetiva la calidad y eficacia del modelo en NER, la métrica estándar recomendada es el Score. Esta métrica es ideal porque evalúa el equilibrio exacto entre la precisión (cuántas clasificaciones positivas fueron correctas) y la sensibilidad o recall (cuántos positivos reales logró detectar el modelo)
## ¿Qué es una arquitectura decoder only?

Una arquitectura decoder-only (de solo decodificador) es una clase de modelo de lenguaje basada en el diseño Transformer que prescinde por completo del componente codificador y utiliza únicamente la fase de decodificación. Estos modelos están especialmente diseñados y enfocados en la generación de texto basado en un contexto previo, siendo ideales cuando el resultado esperado es la creación de una secuencia. Su funcionamiento se caracteriza por los siguientes aspectos clave:

- Procesamiento unidireccional: A diferencia de las arquitecturas basadas en encoders (que leen el contexto en ambas direcciones al mismo tiempo), los modelos de solo decodificador solo miran hacia el pasado.
- Entrenamiento autorregresivo: Aprenden mediante un objetivo continuo de adivinar el futuro. Su entrenamiento consiste netamente en predecir cuál es la próxima palabra o token de una secuencia, basándose estrictamente en los tokens anteriores.
- Generación directa de texto: En las arquitecturas compuestas (encoder-decoder), el codificador primero procesa la entrada y crea un vector intermedio explícito que el decodificador luego utiliza para generar la respuesta. La arquitectura decoder-only elimina este paso intermedio: trata a la secuencia de entrada simplemente como un prefijo condicional y comienza a generar la salida directamente a partir de él, sin crear esa representación compacta intermedia.

El referente absoluto y ejemplo más claro de esta arquitectura en la actualidad es la familia de modelos GPT (Generative Pre-trained Transformer) desarrollados por OpenAI

## ¿Cuáles son las ventajas y desventajas de la arquitectura encoder-decoder vs decoder only?

La comparación entre ambas arquitecturas radica en cómo procesan la información de entrada y cuál es su objetivo principal. Aquí tienes las ventajas y desventajas de cada una: Arquitectura Encoder-Decoder

- Ventajas: Su mayor fortaleza es que "aprovecha mejor" el contexto inicial. Al contar con un codificador bidireccional, procesa toda la oración de entrada en ambas direcciones para lograr una comprensión profunda antes de empezar a generar la respuesta. Por esto, es la arquitectura ideal y más potente para tareas generativas que dependen estrictamente de transformar un texto origen, como la traducción automática o la generación de resúmenes (summarization).
- Desventajas: Su diseño es más complejo y requiere procesar los datos en dos etapas.

Primero crea una representación intermedia de la entrada con el encoder, y luego el decoder debe utilizar una capa especial de atención cruzada (encoder-decoder attention) para conectarse con esa representación y recién ahí poder emitir una salida. Arquitectura Decoder-Only

- Ventajas: Es un diseño excepcionalmente directo, especializado netamente en la generación de secuencias. Al eliminar el codificador, trata cualquier texto de entrada simplemente como un "prefijo" y comienza a generar la salida de forma autorregresiva (prediciendo la próxima palabra) sin necesidad de construir representaciones intermedias explícitas. Esta simplicidad arquitectónica la hizo altamente escalable, convirtiéndose en el estándar de la industria para los Grandes

Modelos de Lenguaje (LLMs) como la familia GPT.

- Desventajas: Su procesamiento es estrictamente unidireccional (solo mira hacia el pasado). Al no contar con un componente bidireccional, el modelo procesa la secuencia inicial paso a paso de izquierda a derecha, perdiéndose la capacidad de analizar el contexto futuro de una palabra dentro de esa misma oración original (algo que los encoders sí hacen muy bien)
## ¿Qué se aprende durante el petraining?

Durante el preentrenamiento (pre-training), al procesar cantidades masivas de texto general sin etiquetar, el modelo aprende una representación sumamente rica del lenguaje y del mundo. Específicamente, el modelo asimila:

- Estructuras del lenguaje: Aprende reglas gramaticales y sintácticas, comprendiendo cómo se construyen las oraciones (por ejemplo, saber qué tipo de palabra encaja sintácticamente en una frase como "El _ a la tienda para comprar...").
- Semántica y contexto: Comprende los matices del significado, el sentido común y el sentimiento o tono de las frases (como deducir lógicamente que si "El hotel estaba sucio, la experiencia fue _", la palabra faltante debe ser negativa).
- Conocimiento general del mundo: Memoriza e incorpora una inmensa cantidad de información factual de diversas áreas. A través de la tarea de adivinar palabras, el modelo aprende sobre historia, geografía, ciencia, operaciones matemáticas básicas, cultura literaria, analogías e incluso refranes populares.

Al adquirir todo este conocimiento previo, los pesos matemáticos de la red neuronal abandonan su estado aleatorio inicial y se posicionan en un punto mucho más cercano al óptimo. Esto es lo que permite que el modelo luego necesite muchos menos recursos y datos para especializarse en una tarea concreta durante la fase de ajuste fino (fine-tuning)

## Que es un MLM?

Un Modelo de Lenguaje Enmascarado (MLM) es un enfoque de preentrenamiento donde la red neuronal aprende a predecir palabras que han sido ocultadas o "enmascaradas" intencionalmente dentro de una oración. El proceso funciona de la siguiente manera:

- Enmascaramiento: Durante el entrenamiento, se selecciona aleatoriamente un porcentaje de las palabras o tokens de la secuencia de texto (por ejemplo, un 15% en arquitecturas como BERT).
- Predicción: El modelo analiza la oración completa e intenta adivinar cuáles son las palabras ocultas basándose en el contexto bidireccional, es decir, observando simultáneamente la información proporcionada por las palabras que están tanto antes como después del hueco.
- Ajuste: La red calcula su pérdida o error matemático evaluando únicamente su desempeño sobre esas palabras que fueron enmascaradas.

Para que el modelo sea robusto y no dependa exclusivamente de ver un símbolo artificial durante el entrenamiento, implementaciones clásicas como BERT aplican una regla sobre ese 15% de tokens seleccionados: el 80% de las veces se ocultan con un token especial llamado [MASK], un 10% de las veces se reemplazan por una palabra completamente aleatoria, y el 10% restante se dejan con la palabra original intacta para que el modelo la confirme. Los representantes más conocidos de este tipo de modelos son BERT, RoBERTa y DeBERTa. Al procesar el contexto en ambas direcciones, son arquitecturas excepcionales para abordar tareas analíticas (como análisis de sentimientos, clasificación de texto o reconocimiento de entidades nombradas), pero no se utilizan para la generación de texto.

## ¿Cómo se pre-entrena Electra?

El preentrenamiento de ELECTRA abandona el enfoque clásico de predecir palabras ocultas y utiliza una arquitectura compuesta por dos redes neuronales que trabajan en conjunto: un Generador y un Discriminador. El proceso paso a paso funciona de la siguiente manera:

1. Generación (MLM pequeño): Primero, se toma una secuencia de texto original y se reemplazan algunos tokens con una máscara ([MASK]). Esta secuencia pasa por el Generador (típicamente un Modelo de Lenguaje Enmascarado pequeño), el cual intenta adivinar qué palabras faltan. El resultado es una oración "corrupta" pero semánticamente plausible. Por ejemplo, si la oración original era "el chef cocinó la comida", el generador podría completarla engañosamente como "el chef comió la comida" .
2. Discriminación (ELECTRA): Esta nueva oración generada se envía al Discriminador, que es el verdadero modelo ELECTRA. Su tarea es actuar como un clasificador binario evaluando todos y cada uno de los tokens de la secuencia. Para cada palabra, ELECTRA debe predecir si es "original" (pertenecía al texto verdadero) o si fue "reemplazada" (fue introducida artificialmente por el generador). Ambas partes se entrenan simultáneamente optimizando una función de pérdida matemática que combina el error del generador (al intentar crear palabras falsas creíbles) y el error del discriminador (al intentar detectar el engaño)
## ¿Cómo se pre-entrena T5?

El modelo T5 (una arquitectura encoder-decoder) se pre-entrena utilizando una técnica principal denominada "span corruption" (corrupción de fragmentos). A diferencia de los modelos de lenguaje enmascarado tradicionales que predicen palabras individuales, este proceso se formula estructuralmente como una tarea de generación de texto donde el objetivo del modelo es aprender a reconstruir el input original. Funciona de la siguiente manera:

1.
2. Se toman oraciones del texto original y se reemplazan secuencias o fragmentos

enteros de palabras contiguas por un único token especial (por ejemplo, el símbolo <X> o <Y>). El modelo procesa esta oración "corrupta" a través de su encoder y luego su decoder debe generar como salida estrictamente el texto que falta, utilizando esos mismos tokens especiales para enmarcar las respuestas. Para ilustrarlo con un ejemplo de las fuentes, si la oración original es "Thank you for inviting me to your party last week", la entrada enmascarada podría ser "Thank you <X> me to your party <Y> week.". El modelo debe aprender a generar la salida exacta: "<X> for inviting <Y> last <Z>". Adicionalmente, este pre-entrenamiento masivo se llevó a cabo utilizando un inmenso conjunto de datos llamado C4 (Colossal Clean Crawled Corpus)

## Mencione los hyperparameters más relevantes de BERT

Los hiperparámetros más relevantes que definen la arquitectura y capacidad del modelo BERT son los siguientes:

- Número de Capas (Num. Layers): Define la profundidad total de la red, es decir, cuántos bloques completos de Transformer Encoder están apilados uno sobre otro.

La versión BERT Base utiliza 12 capas, mientras que la versión BERT Large emplea 24 capas.

- Dimensiones Ocultas (Hidden Dimensions o Hidden Size): Establece el tamaño matemático tanto de los embeddings iniciales como de las representaciones vectoriales internas a lo largo de todo el modelo. Para BERT Base este tamaño es de

768, y para BERT Large aumenta a 1024 dimensiones.

- Número de Cabezas de Atención (Num. Attention Heads): Indica cuántas "cabezas" independientes operan en paralelo dentro de la subcapa de Multi-Head

Attention. Esto le permite al modelo enfocarse en diferentes aspectos del texto al mismo tiempo. BERT Base utiliza 12 cabezas y BERT Large utiliza 16.

- Longitud Máxima de Secuencia (Sequence Length / Max Position Embeddings): Define el límite máximo de tokens (palabras o subpalabras) que el modelo es capaz de procesar en una única entrada de texto. En la arquitectura BERT, este límite está fijado en 512 posiciones.
- Tamaño Intermedio (Intermediate Size): Hace referencia a la cantidad de neuronas que posee la capa oculta dentro de la subred Feed-Forward (la red neuronal tradicional que se encuentra al final de cada bloque Transformer). En la configuración estándar, este valor suele cuadruplicar la dimensión oculta, siendo de 3072 para la versión Base.
- Función de Activación (Activation): Es la operación matemática no lineal utilizada dentro de la red. BERT utiliza específicamente la función GELU.
- Tamaño del Vocabulario (Vocab Size): Determina la cantidad total de tokens únicos que el modelo puede reconocer y generar a partir de su entrenamiento previo. En el caso de BERT (que utiliza el tokenizador WordPiece), el vocabulario consta de

30.522 tokens

## ¿Cómo se preentrena un modelo unidireccional?

Un modelo unidireccional (como los de la familia GPT) se pre-entrena mediante un objetivo autorregresivo conocido como Predicción de la Siguiente Palabra (o NWP, por Next Word Prediction). A diferencia de los modelos bidireccionales que adivinan palabras ocultas en medio de una oración, el proceso en un modelo unidireccional consiste en leer una secuencia de texto y predecir cuál es el próximo token, basándose estrictamente en el contexto de los tokens anteriores (mirando únicamente hacia el pasado). Durante este entrenamiento se utiliza una técnica fundamental llamada Teacher Forcing. Esto significa que, paso a paso, al modelo se lo alimenta con el token correcto real que proviene del texto de entrenamiento (el ground truth) para que intente predecir el siguiente, en lugar de utilizar su propia predicción previa. Alimentarlo siempre con la respuesta correcta estabiliza el entrenamiento y acelera su convergencia. Sin embargo, esta técnica introduce un desafío conocido como Exposure Bias (sesgo de exposición): durante el uso real (inferencia), el modelo ya no cuenta con el texto correcto de ayuda para guiarse, por lo que si comete un error al predecir una palabra, ese error puede propagarse y afectar el resto de la generación de la oración

## Exlicá la arquitectura transformers, como se compara con una RNN, pros

and cons La arquitectura Transformer se basa en el uso exclusivo de mecanismos de atención (como Multi-Head Attention), prescindiendo por completo de la recurrencia. En lugar de leer un texto secuencialmente, permite que cada token calcule su representación matemática prestando atención simultáneamente a todos los demás tokens de la secuencia. Comparación con RNN (Redes Neuronales Recurrentes): Mientras que una RNN procesa la información paso a paso, actualizando un "estado oculto" palabra por palabra, el Transformer procesa todos los elementos de la secuencia en paralelo. Como el Transformer lee todo al mismo tiempo, carece de una noción natural del orden temporal, por lo que requiere sumar obligatoriamente Positional Embeddings a los datos de entrada para saber la posición de cada palabra. Pros de los Transformers (frente a las RNN):

- Velocidad y escalabilidad: Al procesar toda la secuencia en paralelo y no depender del cálculo del paso anterior, los Transformers son significativamente más rápidos de entrenar y permiten escalar a modelos masivos.
- Dependencias a largo plazo: Las RNN forzaban una interacción lineal y sufrían de pérdida de información (vanishing gradient), degradando el contexto de las primeras palabras en textos largos. El mecanismo de atención del Transformer evalúa las relaciones entre palabras directamente, capturando dependencias lejanas sin degradación.
- Eliminación del cuello de botella: En modelos Seq2Seq, la RNN comprimía toda la entrada en un único vector fijo final. La atención permite al decodificador mirar directamente todos los estados generados por el codificador.

Contras de los Transformers (frente a las RNN):

- Ausencia de orden nativo: Las RNN manejan naturalmente el flujo temporal. Los

Transformers necesitan inyectar codificaciones posicionales artificiales porque la red por sí sola no entiende el orden de las palabras.

- Costo computacional en secuencias largas: El mecanismo de self-attention introduce fuertes problemas de escalabilidad computacional a medida que crece la longitud de la secuencia (complejidad cuadrática). Las RNN, en contraste, mantienen una cantidad de parámetros y operaciones por paso que no dependen del largo total de la entrada
## ¿Cómo se pueden evaluar los modelos de lenguaje?

La evaluación de un modelo de lenguaje se aborda mediante dos enfoques complementarios: la evaluación intrínseca y la evaluación extrínseca.

1. Evaluación Intrínseca Se enfoca en medir qué tan bien el modelo entiende el lenguaje en su tarea base general (la generación o predicción de texto), sin aplicarlo a un problema particular.
- La métrica reina para esto es la Perplejidad (Perplexity). Esta métrica calcula qué tan bien las probabilidades predichas por el modelo se alinean con textos de prueba reales.
- Matemáticamente, es la inversa de la probabilidad de la secuencia normalizada por la cantidad de palabras, operando con logaritmos para evitar errores de cálculo por números muy pequeños (underflow).
- Cuanto menor sea la perplejidad, mejor es el modelo. Sin embargo, tiene una advertencia estricta: la perplejidad solo es válida para comparar modelos si ambos comparten el mismo corpus de entrenamiento, el mismo preprocesamiento y exactamente el mismo vocabulario.
2. Evaluación Extrínseca Evalúa el desempeño real del modelo cuando se lo pone a resolver tareas específicas (como Análisis de Sentimientos, Reconocimiento de Entidades Nombradas o Traducción). Como un modelo con baja perplejidad general no siempre garantiza un buen rendimiento en tareas concretas, las métricas cambian dependiendo del objetivo:
- En tareas de clasificación analítica se utilizan métricas como Accuracy (exactitud general), Precisión (cuántos positivos detectados fueron correctos), Sensibilidad o

Recall (cuántos positivos reales detectó), y el F1 Score (el balance armónico entre precisión y recall).

- En tareas puramente generativas, se utilizan métricas especializadas como BLEU o

ROUGE.

3. Evaluación por Preferencias Humanas Con la llegada de modelos generativos modernos (como la familia GPT o LLaMA), el paradigma sumó evaluaciones basadas en validación humana empírica. A los evaluadores se les presentan distintas respuestas generadas por los modelos y deben armar rankings para medir cuál prefieren. Aquí las métricas cambian de enfoque y miden la tasa de victorias (win rate) de un modelo sobre otro, su nivel de factualidad y utilidad, o su prevalencia de toxicidad y "alucinaciones"
## Describí el proceso de MLM.

El proceso de un Modelo de Lenguaje Enmascarado (MLM) consiste en entrenar a una red neuronal para que adivine palabras ocultas utilizando el contexto bidireccional de la oración. De forma resumida y al pie, el proceso es el siguiente:

1.
2.
3.
4. Selección: Durante el preentrenamiento, se elige aleatoriamente un porcentaje de los

tokens del texto de entrada (por ejemplo, un 15% en el caso de BERT). Enmascaramiento: A esos tokens seleccionados se les aplica una transformación: el 80% de las veces se ocultan reemplazándolos por un símbolo especial ([MASK]), el 10% se reemplazan por un token completamente aleatorio, y el 10% restante se deja en su forma original intacta. Predicción: El modelo lee la secuencia alterada e intenta predecir cuáles eran las palabras originales que estaban en las posiciones afectadas, basándose exclusivamente en el contexto que le proporcionan las palabras "visibles" a su alrededor. Ajuste: Para aprender, el modelo calcula su error matemático (o función de pérdida) evaluando su desempeño únicamente sobre esas palabras enmascaradas, ignorando el resto de los tokens de la oración

## ¿Qué es una arquitectura encoder-decoder?

Una arquitectura encoder-decoder es un diseño compuesto por dos partes que trabajan en conjunto, y es la estructura ideal para resolver tareas generativas que requieren transformar un texto de entrada en una nueva salida (como la traducción automática, la generación de resúmenes o la respuesta a preguntas). Su funcionamiento se divide en:

1. Encoder (Codificador): Lee y procesa el texto de entrada original de forma bidireccional, lo que le permite aprovechar al máximo el contexto completo de la oración para crear una representación matemática compacta de toda esa secuencia.
2. Decoder (Decodificador): Recibe esa representación compacta y genera el texto de salida de forma autorregresiva (prediciendo la secuencia paso a paso). En el caso específico de los Transformers, la conexión vital entre ambos componentes se da a través de una subcapa en el decodificador llamada atención cruzada (encoder-decoder attention). Esta capa le permite al decodificador enfocarse directamente en la representación procesada por el codificador para saber qué información del texto original debe utilizar al generar cada nueva palabra
## ¿Qué es el pre-training que relación tiene con el transfer learning?

El pre-entrenamiento (pre-training) es la fase inicial en la que un modelo de lenguaje se entrena utilizando cantidades masivas de texto general sin etiquetar, resolviendo una tarea genérica (como adivinar palabras ocultas o predecir la siguiente palabra de una oración). Durante este proceso a gran escala, el modelo aprende una representación excepcionalmente rica del lenguaje, asimilando estructuras sintácticas, matices semánticos y conocimiento general del mundo. Por otro lado, el Aprendizaje por Transferencia (Transfer Learning) es el paradigma o técnica mediante la cual el conocimiento que un modelo desarrolló para una tarea se reutiliza como punto de partida para resolver una segunda tarea distinta. La relación entre ambos es fundacional: el pre-entrenamiento es justamente el paso que genera ese "conocimiento previo" que hace posible aplicar el Transfer Learning con éxito en el Procesamiento de Lenguaje Natural moderno. La dinámica funciona de la siguiente manera:

1.
2. Evitar empezar desde cero: En lugar de inicializar los pesos matemáticos de una

red neuronal de forma completamente aleatoria para resolver un problema específico, tomamos como base el modelo pre-entrenado. Punto de partida avanzado: Como el modelo base ya pasó por el pre-entrenamiento y comprende cómo funciona el lenguaje de forma general, sus parámetros internos ya se encuentran posicionados en un lugar mucho más cercano al punto óptimo necesario para resolver tu problema.

3. Transferencia (Fine-Tuning): Finalmente, aplicamos Transfer Learning ajustando finamente ese modelo general para adaptarlo a nuestra tarea específica (por ejemplo, clasificar textos o extraer entidades) con nuestros propios datos. Gracias a esta estrecha relación, maximizamos el conocimiento preexistente del modelo, logrando reducir drásticamente el tiempo de entrenamiento, los recursos computacionales necesarios y, fundamentalmente, la cantidad de datos etiquetados a mano que necesitamos recolectar para lograr que el modelo tenga un excelente rendimiento en tareas nuevas
## En qué capas se centra el PT y en cuales el FT En el contexto del paradigma de transferencia de conocimiento, el enfoque de lo que aprende el modelo se distribuye en diferentes niveles de la red neuronal de la siguiente manera:

- Pre-entrenamiento (PT): Su conocimiento fundacional se consolida fuertemente en las primeras capas (o capas inferiores) de la red. Al analizar inmensas cantidades de texto general, estas capas bajas se encargan de aprender características estructurales amplias, centrándose en aspectos sintácticos, como la gramática, la estructura de las oraciones y las etiquetas de partes del discurso (POS tags).
- Ajuste fino (Fine-Tuning - FT): Afecta de manera mucho más significativa a las capas más profundas (o capas superiores). Mientras las primeras capas conservan esa base sintáctica general, las capas altas se encargan de asimilar las características más específicas para la nueva tarea, enfocándose fuertemente en la captura de información semántica, es decir, el significado profundo y las particularidades relevantes al dominio del problema. Además, durante el Fine-Tuning es donde se acopla una nueva capa final especializada (como una capa densa o un clasificador

Softmax) que interactúa con estas capas profundas para emitir la predicción de la tarea

## Que tipos de preentrenamiento conoces?

Según el material de las clases, existen principalmente los siguientes tipos de preentrenamiento:

- Modelo de Lenguaje Enmascarado (MLM): Consiste en enmascarar u ocultar palabras dentro de una oración para que el modelo intente predecirlas utilizando el contexto. Es el enfoque característico de modelos bidireccionales como BERT.
- Predicción de la Siguiente Palabra (NWP - Next Word Prediction): Se trata de un entrenamiento autorregresivo donde el objetivo es adivinar la próxima palabra en una secuencia, basándose únicamente en el texto anterior. Es el método fundamental utilizado por modelos como GPT.
- Predicción de la Siguiente Oración (NSP - Next Sentence Prediction): El modelo aprende a determinar si una oración específica es la continuación lógica de otra dentro de un documento. También fue utilizado en el entrenamiento original de BERT.
- Discriminativo: En lugar de intentar predecir y generar la palabra faltante, el modelo actúa como un clasificador evaluando cada token. Aprende a detectar errores identificando si las palabras de una secuencia son las originales o si fueron introducidas/reemplazadas artificialmente por un pequeño generador MLM. Este es el caso del modelo ELECTRA.
- Corrupción de fragmentos y eliminación de ruido (Span Corruption / Denoising): Utilizado en arquitecturas Encoder-Decoder. Se ofusca o corrompe el texto de entrada aplicando técnicas como el enmascaramiento de grupos de palabras (span corruption), la eliminación de tokens, o incluso la permutación de oraciones y rotación de documentos. El objetivo del modelo es reconstruir y generar el texto original exacto como salida. Ejemplos de esto son T5 y BART
## ¿Qué información incorpora el pre entrenamiento?

Durante la fase de preentrenamiento, al procesar cantidades masivas de texto general sin etiquetar, el modelo incorpora una representación sumamente rica y profunda del lenguaje y del mundo. Específicamente, asimila los siguientes tipos de información:

- Estructuras sintácticas y gramaticales: El modelo aprende las reglas subyacentes de cómo se construyen las oraciones, identificando partes del discurso (POS tags) y las relaciones sintácticas estructurales entre las palabras.
- Semántica y contexto: Comprende el significado de las palabras, sus matices (nuances semánticos) y cómo el contexto que las rodea puede alterar por completo su intención.
- Conocimiento general y factual del mundo: A través de la tarea de adivinar palabras, el modelo actúa como una base de datos comprimida, memorizando información concreta sobre historia, ciencia, geografía o cultura general (por ejemplo, aprende a completar datos históricos como que en 1789 se tomó la Bastilla, o datos científicos como la temperatura de ebullición del agua).
- Sentido común y razonamiento básico: Incorpora habilidades analíticas implícitas, como la capacidad de resolver analogías (entender que un león es a una manada lo que un lobo es a una jauría), completar refranes populares, o incluso realizar operaciones matemáticas simples.

Al adquirir toda esta información fundacional durante el preentrenamiento, el modelo reduce drásticamente la cantidad de datos especializados que necesitará aprender desde cero durante la fase de ajuste fino (fine-tuning)

## Explicá conceptualmente el paradigma PT y FT

El paradigma de Preentrenamiento (PT) y Ajuste Fino (FT) es el núcleo del Aprendizaje por Transferencia (Transfer Learning) en el desarrollo de modelos de lenguaje modernos. Se divide en dos etapas complementarias:

1. Preentrenamiento (PT): Es la etapa inicial y generalista. El modelo se entrena sobre cantidades masivas de texto sin etiquetar resolviendo tareas genéricas (como adivinar palabras ocultas o predecir la siguiente palabra). Aquí el modelo aprende la base del lenguaje: sintaxis, semántica y conocimiento del mundo. Al finalizar, sus parámetros matemáticos dejan de ser aleatorios y se ubican en un punto muy avanzado.
2. Ajuste Fino (FT): Se toma ese modelo base ya "educado" y se lo entrena nuevamente, pero esta vez con tus datos especializados y etiquetados para resolver una tarea muy concreta (como clasificar sentimientos o extraer entidades). Conceptualmente, es como si a una persona primero se le enseñara a leer, hablar y comprender el mundo en general (PT), para luego darle un entrenamiento mucho más corto y específico que lo convierta en abogado o médico (FT). Gracias a este paradigma, al iniciar el Ajuste Fino desde un punto tan avanzado, se reducen drásticamente los costos computacionales, el tiempo de entrenamiento y la necesidad de recolectar enormes cantidades de datos etiquetados a mano
## ¿Qué son los adapters?

Los Adapters (o adaptadores) son pequeños módulos neuronales entrenables que se insertan directamente dentro de las capas de un modelo pre-entrenado. Su objetivo principal es ofrecer una alternativa eficiente frente a las grandes limitaciones del Fine-Tuning tradicional. Normalmente, para adaptar un modelo a una nueva tarea, se deben modificar todos y cada uno de sus millones (o billones) de parámetros internos. Esto requiere inmensos recursos computacionales, mucho tiempo, y conlleva un gran riesgo de sobreajuste (overfitting) si se tienen pocos datos. Al usar adapters, el paradigma cambia:

1. Se congelan (se dejan intactos) todos los pesos matemáticos originales del modelo
2. base. Durante el ajuste fino, solo se entrenan los parámetros de estos pequeños módulos insertados para que aprendan la tarea específica. Principales ventajas de usar Adapters:
- Eficiencia extrema: Al entrenar una fracción mínima de los parámetros totales, se reducen drásticamente los requerimientos de memoria, cómputo y tiempo de entrenamiento.
- Menos sobreajuste: Como hay muchos menos parámetros que aprender, el riesgo de overfitting baja considerablemente, lo que es ideal cuando tienes un conjunto de datos pequeño.
- Modularidad y Aprendizaje Multi-tarea: En lugar de tener que copiar y guardar un modelo gigante entero para cada tarea distinta, puedes mantener un único modelo base en memoria e ir "enchufándole" distintos adapters livianos dependiendo de lo que necesites resolver en ese momento.
- Preservación del conocimiento: Garantizan que el modelo no pierda ni "olvide" el conocimiento general y profundo que adquirió durante su fase de preentrenamiento.

Existen diferentes formas de integrarlos en la arquitectura, como los adaptadores en serie, los adaptadores en paralelo, y variaciones muy populares en la actualidad orientadas a grandes modelos de lenguaje, como LoRA (Low-Rank Adaptation)

## ¿Qué es LORA?

LoRA (Low-Rank Adaptation) es una técnica avanzada y muy popular dentro de la familia de los adapters, diseñada específicamente para realizar el ajuste fino (fine-tuning) de Grandes Modelos de Lenguaje (LLMs) de una forma extremadamente eficiente. Su funcionamiento se basa en el siguiente proceso:

1. Congelamiento: Primero, se congelan por completo todos los parámetros y pesos matemáticos originales del modelo pre-entrenado.
2. Inyección de matrices: En cada capa de la red neuronal (generalmente en la arquitectura Transformer), se introducen dos matrices nuevas y mucho más pequeñas (de bajo rango).
3. Entrenamiento focalizado: En lugar de modificar la gigantesca matriz de pesos original, el modelo solo entrena y ajusta estas dos matrices pequeñas. Estas logran "aproximar" los cambios que el modelo entero habría necesitado aprender. Durante la generación de texto, simplemente se combinan matemáticamente los cálculos de los pesos originales fijos con los de estas nuevas matrices. Sus principales ventajas son:
- Reducción drástica de costos: Al entrenar una fracción mínima de los parámetros totales, reduce significativamente el tiempo de entrenamiento y los recursos computacionales necesarios.
- Mitigación del sobreajuste: Minimiza el riesgo de overfitting, lo cual es ideal cuando se dispone de un conjunto de datos pequeño para enseñar una tarea específica.
- Agilidad y multi-tarea: Permite cambiar rápidamente entre distintas tareas. Solo necesitas guardar y cargar estas pequeñas matrices ligeras según lo que quieras resolver, manteniendo un único modelo gigante base intacto en la memoria
## ¿Qué es un encoder-decoder?

Un modelo encoder-decoder es un diseño arquitectónico compuesto por dos partes principales, ideal para abordar tareas generativas donde se necesita transformar una secuencia de entrada en una nueva secuencia de salida. Su funcionamiento se divide en los siguientes componentes:

- El Encoder (Codificador): Es el encargado de procesar el texto de entrada original.

Al leer la información de forma bidireccional, logra "aprovechar mejor" todo el contexto de la oración. Su objetivo final es procesar esos datos para crear una representación compacta de la entrada.

- El Decoder (Decodificador): Toma la representación compacta creada por el codificador y la utiliza para producir la salida final. A diferencia del codificador, el decodificador opera de forma autorregresiva, es decir, genera el nuevo texto paso a paso.

Para que ambas partes trabajen en conjunto dentro de un modelo Transformer, el decodificador cuenta con una subcapa especial conocida como atención encoder-decoder. Este mecanismo actúa como puente, permitiendo que el decodificador se enfoque directamente en los resultados del codificador para saber qué información de la entrada original debe utilizar al generar cada nueva palabra. Esta estructura en dos etapas es la arquitectura ideal para resolver tareas que implican transformar texto, tales como la traducción automática, la generación de resúmenes (summarization) y los sistemas de respuesta a preguntas (question answering). Un referente moderno de este diseño es el modelo T5, el cual aborda todas las tareas de NLP estrictamente como problemas de generación de texto Como se preentrena cada arquitectura encoder only, encoder-decoder, decoder only. Menciona un método para cada una. Cada arquitectura de la familia Transformer se pre-entrena con un objetivo distinto, alineado a su estructura y propósito principal:

- Encoder-only (solo codificador): Se pre-entrena habitualmente mediante el método de Modelo de Lenguaje Enmascarado (MLM). En este enfoque, se ocultan o enmascaran aleatoriamente ciertas palabras (o tokens) de la oración de entrada, y el objetivo del modelo es intentar predecir cuáles eran esas palabras originales basándose en el contexto bidireccional (las palabras "visibles" que están tanto antes como después del hueco). Un ejemplo clásico que utiliza este método es BERT.
- Encoder-decoder (codificador-decodificador): Un método característico para esta arquitectura es la corrupción de fragmentos (span corruption) o Denoising. Aquí, el texto de entrada se ofusca o corrompe intencionalmente (por ejemplo, enmascarando bloques enteros de palabras, eliminando tokens o permutando oraciones). El objetivo del modelo es procesar esa entrada alterada y aprender a reconstruir y generar el texto original exacto como salida. Este enfoque se utiliza en modelos como T5 y BART.
- Decoder-only (solo decodificador): Se pre-entrena mediante la Predicción de la

Siguiente Palabra (NWP), un proceso puramente autorregresivo. El modelo analiza una secuencia de entrada y su objetivo es adivinar estrictamente cuál será la próxima palabra o token, basándose de forma exclusiva en el contexto de los tokens anteriores. Este es el método fundamental de los modelos generativos modernos como la familia GPT

## ¿Qué son las capacidades emergentes?

Las capacidades emergentes son habilidades que el modelo no fue programado explícitamente para resolver. En el contexto de los Grandes Modelos de Lenguaje (LLMs), estas habilidades surgen de manera impredecible a medida que el modelo crece en tamaño y no se pueden extrapolar de forma lineal a partir del rendimiento que tienen los modelos más pequeños. Gracias a esta emergencia, al escalar masivamente su cantidad de parámetros, un modelo adquiere de pronto la capacidad de generalizar automáticamente y realizar tareas complejas para las que no fue entrenado directamente, como programar código, traducir idiomas, resumir textos, seguir instrucciones detalladas, diseñar experimentos o entender datos estructurados. Aunque este fenómeno todavía es en gran parte inexplicado y ocurre sin necesidad de realizar cambios estructurales en la arquitectura de la red, las principales teorías plantean que operar a mayor escala permite una mejor memorización y un mejor manejo de tareas complejas. Se cree que al tener una mayor profundidad (más cantidad de capas), el modelo logra realizar razonamientos lógicos de múltiples pasos y consigue almacenar el conocimiento del mundo de una forma mucho más comprimida y eficaz

## ¿Cómo surgen las capacidades emergentes?

Aunque su origen exacto sigue siendo en gran parte inexplicado, sabemos que estas capacidades surgen al escalar masivamente los modelos (aumentando sus parámetros, datos y cómputo), y su aparición es impredecible y no se puede extrapolar de forma lineal a partir de lo que hacen modelos más pequeños. Las principales teorías e hipótesis en la comunidad científica sobre cómo y por qué surgen al aumentar la escala plantean lo siguiente:

- Mejor memorización y manejo de complejidad: Operar a una escala masiva simplemente le permite al modelo memorizar mejor y tener la capacidad de procesar tareas mucho más complejas.
- Profundidad para el razonamiento: Se cree que el razonamiento lógico de múltiples pasos no es posible en redes pequeñas, sino que requiere obligatoriamente una cierta profundidad (un número mínimo de capas) en la red neuronal para poder desarrollarse.
- Compresión de conocimiento: Al tener una cantidad gigantesca de parámetros, el modelo logra almacenar todo el conocimiento del mundo de una manera sumamente comprimida y eficaz
## Cual es la diferencia entre un modelo base y un instructioned tuned?

La diferencia principal entre un modelo base y un modelo instruction-tuned (afinado por instrucciones) radica en su objetivo de entrenamiento y en su capacidad para actuar como un asistente útil.

1. Modelo Base:
- Es el resultado de la fase inicial de preentrenamiento masivo, donde consume enormes cantidades de texto general sin etiquetar.
- Su objetivo principal es puramente estadístico: predecir la siguiente palabra en una secuencia basándose en los patrones del lenguaje.
- No está entrenado explícitamente para comprender o ejecutar instrucciones. Si le haces una pregunta directa, es posible que en lugar de responderla, intente autocompletarla con más preguntas, ya que su instinto es continuar el texto.
- Como su función matemática no está alineada con ser un asistente, puede generar respuestas que no cumplen con la intención del usuario, o incluso producir contenido tóxico y dañino.
- Para que un modelo base resuelva una tarea, suele ser necesario usar técnicas como few-shot learning, dándole ejemplos dentro del prompt para que detecte y repita el patrón.
2. Modelo Instruction-Tuned (Asistente):
- Toma como punto de partida a un modelo base y lo somete a una etapa adicional de entrenamiento llamada Instruction Tuning (o Supervised Fine-Tuning).
- En esta fase, se le enseña explícitamente a entender y seguir directivas humanas utilizando conjuntos de datos que contienen instrucciones ("resume este texto", "traduce este código") y sus salidas esperadas.
- Su objetivo central es alinear el comportamiento del modelo con la intención del usuario, preparándolo para ser un asistente fidedigno, honesto y seguro (no tóxico).
- Mejora drásticamente el rendimiento zero-shot, lo que significa que puedes darle una orden directa y el modelo la ejecutará correctamente sin necesidad de que le des ejemplos previos de cómo hacerlo
## ¿Qué tareas son mas apropiadas para un modelo base y un instruction tuned?

Como repasamos en la pregunta anterior, la principal diferencia radica en cómo están entrenados para interactuar con el usuario. Debido a esto, cada modelo brilla en escenarios distintos: Para un modelo base, las tareas más apropiadas son:

- Generación de texto continuo: Dado que su objetivo principal es predecir la siguiente palabra basándose en la estadística del lenguaje, son excelentes para autocompletar un texto de manera fluida y coherente.
- Reconocimiento de patrones (Few-Shot Learning): Aunque no entienden órdenes directas, pueden resolver tareas si dentro de la entrada (prompt) se les proporcionan varios ejemplos explícitos de lo que deben hacer para que detecten el patrón y lo repitan.
- Servir como cimiento para el Fine-Tuning: Son el punto de partida ideal para aplicar aprendizaje por transferencia. Se pueden reentrenar con datos etiquetados específicos para crear modelos altamente especializados en tareas analíticas como clasificación de secuencias, análisis de sentimientos, o reconocimiento de entidades (NER).

Por otro lado, un modelo afinado por instrucciones (instruction-tuned) está optimizado para actuar como un asistente alineado con las preferencias humanas. Sus tareas más apropiadas incluyen:

- Ejecución directa (Zero-Shot): Son ideales para resolver tareas recibiendo únicamente una orden clara en lenguaje natural, sin necesidad de que el usuario le provea ejemplos previos de cómo hacerlo.
- Asistencia multipropósito: Son la herramienta adecuada para tareas prácticas y de ayuda al usuario, como resumir textos, responder preguntas (abiertas o sobre un texto específico), traducir código o idiomas, realizar lluvias de ideas (brainstorming), reescribir correos y extraer información estructurada de un documento.
- Mantenimiento de diálogos (Chat): Están preparados para sostener una conversación interactiva, siguiendo restricciones explícitas dadas por el usuario, intentando ser fidedignos, honestos, y evitando generar respuestas tóxicas o dañinas
## Describa los tres pasos de entrenamiento para generar un asistente (e.g.,

chatgpt) Para generar un asistente como ChatGPT, el entrenamiento se divide en tres pasos principales basados en la técnica de Aprendizaje por Refuerzo a partir de Retroalimentación Humana (RLHF):

1.
2.
3. Ajuste Fino Supervisado (SFT - Supervised Fine-Tuning): Se recolectan datos de

demostración donde humanos escriben las respuestas ideales para diferentes instrucciones. El modelo base preentrenado se ajusta con estos ejemplos para que aprenda el formato de diálogo y empiece a seguir órdenes. Entrenamiento del Modelo de Recompensa (Reward Model): Se le da un mismo prompt al modelo para que genere varias respuestas distintas. Luego, evaluadores humanos las ordenan (rankean) de mejor a peor. Con estos rankings, se entrena un segundo modelo (generalmente más pequeño) cuya única función es predecir qué respuesta preferiría un humano y asignarle una puntuación. Aprendizaje por Refuerzo (RL con PPO): Se optimiza el modelo original usando aprendizaje por refuerzo (específicamente el algoritmo PPO o Proximal Policy Optimization). El modelo genera una respuesta, el Reward Model del paso anterior la califica, y el modelo ajusta sus parámetros para maximizar esa recompensa, volviéndose más seguro, útil y alineado con la intención del usuario

## ¿Qué datos necesito para el entrenamiento de instruction tuning?

Para el entrenamiento de Instruction Tuning (o Ajuste por Instrucciones), necesitas un conjunto de datos específico compuesto por pares de instrucciones dadas por un usuario (prompts) y sus correspondientes respuestas ideales esperadas. A diferencia de la fase inicial de preentrenamiento, una de las características clave de esta etapa es que requiere una cantidad de datos relativamente pequeña para lograr alinear el comportamiento del modelo y convertirlo en un asistente. Por ejemplo, InstructGPT utilizó poco más de 12.000 ejemplos para su fase principal de ajuste supervisado. Los datos que necesitas deben cumplir con las siguientes características:

- Estructura de pares: Cada ejemplo dentro del dataset debe estar estructurado típicamente como una tupla que contiene la instrucción del usuario, un texto de entrada o contexto opcional, y la salida generada correcta ({user instruction, input}, output).
- Alta Diversidad: Es fundamental que el conjunto de datos cubra una amplia variedad de casos de uso para que el modelo aprenda a generalizar múltiples tipos de peticiones. Según la distribución típica, debes incluir tareas de: generación de texto pura, respuesta a preguntas (QA abiertas y cerradas), lluvia de ideas (brainstorming), estilo de chat, reescritura de textos, resúmenes, extracción de información y clasificación.
- Alta Calidad: Dado que el volumen de datos es mucho menor, la calidad de las respuestas enseñadas es crucial. Estas respuestas son las que le enseñan al modelo a ser explícitamente fidedigno, útil, honesto y a no generar contenido tóxico.
## ¿De dónde provienen estos datos?

Los datasets para Instruction Tuning se construyen principalmente a través de tres fuentes:

1. Anotadores humanos (Labelers): Especialistas o contratistas que redactan a mano tanto las peticiones originales como las respuestas de alta calidad esperadas (enfoque human-crafted).
2. Uso real de clientes: Prompts reales recopilados de usuarios que interactúan con una API o sistema en producción (resguardando la privacidad), a los cuales luego se les redacta una respuesta ideal.
3. Datos sintéticos (Generados por modelos): Se utiliza un modelo más grande y avanzado (como GPT-4) actuando como "maestro" para autogenerar miles de instrucciones variadas junto con sus respuestas. Este conjunto de datos sintético se utiliza luego para afinar un modelo "estudiante" más pequeño. Esta técnica se volvió extremadamente popular para crear modelos open-source como Alpaca o Vicuna
## ¿Cómo generaría datos sintéticos para un entrenamiento de instruction tuning para un modelo 8B?

Para generar datos sintéticos para entrenar un modelo de 8B (que actuaría como un modelo "chico" o estudiante), se utiliza una técnica conocida como Destilación de Datos (Data Distillation). De forma resumida, el proceso es el siguiente:

1. Utilizar un modelo "maestro": Se recurre a un modelo de lenguaje mucho más grande y avanzado (como GPT-4 o ChatGPT) para que actúe como el generador del contenido.
2. Generación a partir de semillas: Se le provee al modelo maestro un conjunto pequeño de instrucciones iniciales (seed instructions) y se le pide que autogenere miles de nuevas instrucciones similares junto con sus respuestas ideales correspondientes.
3. Ajuste Fino (Fine-Tuning): Una vez que el modelo maestro construyó este gran dataset sintético (pares de instrucción y salida), lo utilizas para afinar tu modelo de 8B preentrenado mediante Instruction Tuning. De esta forma, el modelo estudiante de 8B aprende a imitar el comportamiento, el formato y la calidad de las predicciones del modelo gigante. Un ejemplo clásico de esto es el modelo Alpaca, que se afinó utilizando 52.000 datos generados sintéticamente por GPT
## Describa el proceso de entrenamiento instruction tuning

El proceso de Instruction Tuning (o ajuste por instrucciones) es una fase fundamental de Ajuste Fino Supervisado (SFT) que sirve para transformar a un modelo base estadístico en un asistente útil. De forma breve, el proceso consta de los siguientes pasos:

1.
2.
3. Recolección de demostraciones: Se ensambla un conjunto de datos que contiene

pares de {instrucción del usuario, respuesta ideal esperada}. Es vital que estos datos cubran múltiples tareas diversas (como resumir, escribir código, traducir o extraer información). Entrenamiento Supervisado: El modelo base preentrenado lee la instrucción y se le enseña a replicar exactamente la respuesta de demostración mediante aprendizaje supervisado. Cálculo del error (Loss): Un detalle técnico clave durante este proceso es que se enmascara el prompt. Esto significa que la función de pérdida (el error matemático del modelo) se calcula y optimiza únicamente basándose en los tokens de la respuesta generada, ignorando las palabras de la instrucción original. Como resultado de este entrenamiento mult-itarea, el modelo aprende a seguir directivas explícitas y mejora drásticamente su capacidad para resolver instrucciones nuevas que nunca antes había visto (zero-shot)

## ¿Qué es in-context learning?

El In-context learning (aprendizaje en contexto) es un paradigma donde, en lugar de modificar o reentrenar internamente un modelo para resolver un problema específico (como se hace en el fine-tuning), se adapta la tarea al modelo. Sus características principales son:

- Uso directo (out-of-the-box): El modelo se utiliza tal como fue entrenado originalmente, sin actualizar sus pesos o parámetros matemáticos, proporcionándole simplemente un input mínimo.
- Interacción natural: La forma de pedirle que resuelva el problema es intuitiva, interactuando con el modelo directamente en lenguaje natural a través del prompt.

En la práctica, esto significa que el modelo aprende a resolver la tarea "en el momento" basándose únicamente en el contexto, las instrucciones y los ejemplos que le proporcionas en tu texto de entrada.

## ¿Qué es zero-shot, few-shot?

Ambos son enfoques dentro del paradigma de aprendizaje en contexto (in-context learning), donde le pedimos al modelo que resuelva una tarea a través del prompt sin modificar sus parámetros internos:

- Zero-shot (Cero ejemplos): Consiste en darle al modelo una instrucción directa para que resuelva una tarea, sin proporcionarle ningún ejemplo previo de cómo debe hacerlo. Como repasamos antes, el entrenamiento por instrucciones (instruction tuning) es justamente lo que mejora drásticamente esta capacidad, permitiendo que el modelo entienda tu orden a la primera.
- Few-shot (Pocos ejemplos): Consiste en incluir dentro del prompt algunos ejemplos resueltos (pares de entrada y salida ideal) antes de darle tu petición real. Esto le sirve al modelo para identificar el patrón, el formato o el tono que esperás y replicarlo. Es la estrategia fundamental que utilizan los modelos base (no entrenados como asistentes, como el GPT-3 original) para lograr resolver tareas
## ¿Qué son las alucinaciones?

Las alucinaciones en los Grandes Modelos de Lenguaje (LLMs) ocurren cuando el modelo genera un texto que es semántica o sintácticamente plausible (es decir, suena fluido, coherente y aparentemente arraigado en el contexto), pero que en realidad es factualmente incorrecto o carece de sentido. A simple vista, suele ser muy difícil darse cuenta de que esta información es errónea debido a lo natural y convincente que parece la respuesta generada. Existen principalmente dos tipos de alucinaciones:

- Alucinación Intrínseca: Ocurre cuando el contenido generado por el modelo contradice directamente la información de la fuente original. Un ejemplo clásico es cuando el modelo inventa o representa erróneamente fechas o eventos factuales concretos.
- Alucinación Extrínseca: Se da cuando el contenido generado añade detalles que no se pueden verificar ni contradecir con la fuente proporcionada. Es decir, el modelo incorpora información adicional que simplemente no estaba presente en tu texto de entrada.

Para entender y evaluar estas alucinaciones, es importante distinguir entre dos conceptos clave: la fidelidad (qué tanto el modelo se adhiere estrictamente al contenido de la fuente, con el objetivo de minimizar la invención) y la facticidad (qué tan alineado está el texto generado con hechos reales o el conocimiento actual del mundo).

## ¿Por qué ocurren las alucinaciones?

Las causas detrás de este fenómeno se agrupan en dos categorías principales:

1. Causas impulsadas por los datos: Las alucinaciones pueden surgir simplemente porque existen errores o información falsa en los datos masivos con los que el modelo fue entrenado. También pueden originarse debido a la propia naturaleza de ciertas tareas de generación de lenguaje que fomentan la divergencia y la creatividad, lo que lleva al modelo a "inventar" .
2. Influencias del modelado y entrenamiento: Varios factores técnicos contribuyen a este problema, tales como un aprendizaje de representación imperfecto, errores matemáticos durante el proceso de decodificación, el sesgo de exposición (exposure bias) introducido durante la fase de entrenamiento, y las limitaciones del conocimiento fijo e inmutable que el modelo tiene sobre el mundo
## ¿Cómo se mitigan las alucinaciones?

Para mitigar las alucinaciones en los modelos de lenguaje, se emplean diferentes enfoques técnicos y metodológicos enfocados en los datos, el entrenamiento y la arquitectura:

- Mejorar la calidad de los datos: Dado que gran parte de las alucinaciones se originan por errores o falsedades presentes en los datos de entrenamiento masivos, refinar y elevar la calidad de estos conjuntos es un paso esencial.
- Mejorar los modelos: Implica corregir vulnerabilidades durante el entrenamiento, como perfeccionar el aprendizaje de las representaciones, mejorar los procesos matemáticos de decodificación y mitigar sesgos de exposición.
- Aprendizaje por refuerzo mediante modelos de recompensa: Utilizar técnicas de alineación para guiar el comportamiento del modelo, recompensando las respuestas veraces y penalizando aquellas que inventan hechos.
- Atención y condicionamiento por la fuente: Ajustar los mecanismos de atención para que el modelo se vea forzado a ceñirse estrictamente al texto o contexto proporcionado, maximizando su fidelidad a la fuente y evitando que su natural divergencia creativa genere detalles no verificables.
- Integración de información externa (RAG): Proveer al modelo acceso a bases de conocimiento externas. Al ampliar el contexto con documentos precisos (como información interna de una empresa o bases de datos específicas) utilizando técnicas como la Generación Aumentada por Recuperación (RAG), se mejora drásticamente la factualidad de las respuestas al anclar las predicciones del modelo a información real y recuperada en el momento
## ¿Cómo se pasa de un modelo base a un asistente?

Para transformar un modelo base estadístico en un asistente útil, se aplica un proceso de alineación que típicamente consta de tres etapas principales:

1. Ajuste Fino Supervisado (SFT o Instruction Tuning): Se ajusta el modelo base utilizando ejemplos de alta calidad (pares de instrucción y respuesta ideal) para enseñarle explícitamente a comprender y seguir órdenes.
2. Modelo de Recompensa (Reward Model): Se entrena un modelo secundario para calificar y rankear diferentes respuestas generadas, basándose en lo que los humanos prefieren (fidelidad, utilidad, no toxicidad).
3. Aprendizaje por Refuerzo (RLHF): Se utiliza la calificación del modelo de recompensa para optimizar el comportamiento del asistente mediante algoritmos como PPO. Nota: Actualmente también existen técnicas más modernas como DPO (Direct Preference Optimization), que simplifican este proceso optimizando las preferencias directamente sin necesidad de entrenar el Modelo de Recompensa del paso 2
## ¿Qué tipo de prompts funcionan en los modelos base, zero-shot o few-shot?

Por que? En los modelos base, el tipo de prompt que funciona para resolver tareas es el few-shot (con pocos ejemplos). Esto ocurre porque los modelos base se entrenan con un objetivo puramente estadístico enfocado en predecir la siguiente palabra, por lo que no están capacitados explícitamente para comprender y ejecutar instrucciones directas. Si se les da un prompt en formato zero-shot (una orden directa sin ejemplos previos), el modelo no interpretará que debe cumplir una tarea, sino que intentará simplemente autocompletar el texto. Al utilizar un prompt few-shot, le brindamos al modelo un contexto que incluye ejemplos previos de la tarea resuelta. Gracias a que estos modelos son excelentes para detectar y repetir secuencias, logran identificar el patrón del texto y generar la respuesta correcta simplemente continuando la estructura que les proporcionaste

## ¿Qué es el instruction tuning como se diferencia de PT?

El Instruction Tuning (o ajuste por instrucciones) es una etapa de entrenamiento (parte del Ajuste Fino Supervisado) que sirve para alinear el comportamiento del modelo con la intención del usuario, enseñándole explícitamente a seguir directivas para que actúe como un asistente seguro, fidedigno y honesto. Las principales diferencias respecto al Preentrenamiento (PT) son:

- Objetivo principal: El PT tiene un objetivo estadístico y general, enfocado principalmente en predecir la siguiente palabra para que el modelo incorpore el conocimiento del mundo y la estructura del lenguaje. Por el contrario, el instruction tuning tiene el objetivo de que el modelo deje de simplemente "autocompletar" y aprenda a responder a órdenes concretas, evitando generar respuestas tóxicas o dañinas.
- Volumen y tipo de datos: El PT se realiza mediante aprendizaje no supervisado, consumiendo cantidades masivas de texto general sin etiquetar. El instruction tuning es un proceso supervisado que requiere una cantidad de datos mucho menor, basados en demostraciones específicas (pares de instrucciones del usuario y sus respuestas ideales).
- Resultados de ejecución: Un modelo base (solo con PT) no está preparado para interactuar con humanos siguiendo órdenes directas. Al aplicar instruction tuning, se mejora drásticamente el rendimiento zero-shot, lo que significa que el modelo adquiere la capacidad de ejecutar tareas a la primera, basándose únicamente en la instrucción recibida
## ¿Cómo se pueden coleccionar los datos para Instruction Tuning?

Los datos para Instruction Tuning se pueden coleccionar principalmente a través de tres enfoques:

1. Anotadores humanos (Labelers o Human-crafted): Personas contratadas que redactan a mano tanto las instrucciones (prompts) como las respuestas ideales esperadas para garantizar una alta calidad.
2. Clientes o usuarios reales (Customers o User-shared): Se recopilan peticiones reales que los usuarios hacen al sistema (por ejemplo, a través de una API), a las cuales luego se les genera o asigna una respuesta óptima.
3. Generación sintética (Model-generated): Se utiliza un modelo de lenguaje más grande y avanzado (que actúa como modelo "maestro") para generar automáticamente miles de instrucciones y sus respectivas respuestas. Estos datos autogenerados se utilizan luego para entrenar a un modelo más pequeño ("estudiante")
## Se pueden coleccionar sintéticamente? Como?

Sí, los datos para Instruction Tuning se pueden generar de forma sintética mediante una técnica conocida como Destilación de Datos o Destilación de Modelos. El proceso para hacerlo consta de los siguientes pasos:

1.
2. Utilizar un modelo "maestro": Se recurre a un Modelo de Lenguaje de Gran Escala

(LLM) muy avanzado y potente, como GPT-4 o ChatGPT. Generación a partir de semillas (seed instructions): Se le proporciona al modelo maestro un pequeño conjunto inicial de instrucciones humanas. A partir de estas, se

3. le pide que autogenere miles de instrucciones nuevas y variadas, junto con sus correspondientes respuestas ideales (outputs). Entrenamiento del modelo "estudiante": Una vez generado este gran conjunto de datos sintético (pares de instrucción y respuesta), se utiliza para afinar (fine-tune) un modelo más pequeño mediante Instruction Tuning. De esta manera, se logra transferir el conocimiento del modelo gigante al modelo más pequeño, haciendo que el "estudiante" aprenda a imitar el comportamiento y las predicciones del "maestro" de una forma mucho más eficiente y económica
## ¿Qué es el RLHF?

¿Para qué sirve? El RLHF (Aprendizaje por Refuerzo a partir de Retroalimentación Humana) es una técnica de entrenamiento que formula el ajuste de un modelo de lenguaje como un problema de aprendizaje por refuerzo. Funciona optimizando el modelo a través de un Modelo de Recompensa (Reward Model); este último evalúa las respuestas generadas basándose en lo que los humanos prefieren, y luego un algoritmo (típicamente PPO) ajusta los parámetros del asistente para que aprenda a maximizar esa puntuación.

## ¿Para qué sirve?

Sirve fundamentalmente para alinear el comportamiento del modelo con las intenciones y valores humanos. Sus objetivos principales son:

- Crear asistentes útiles: Es el paso definitivo (luego del Instruction Tuning) para consolidar a un modelo base estadístico como un asistente conversacional capaz de seguir directivas complejas (como en ChatGPT o LLaMA 2).
- Mejorar la seguridad: Actúa como una técnica de entrenamiento adaptativo muy efectiva para reducir la toxicidad, mitigar sesgos sociales y evitar que el modelo genere respuestas dañinas.
- Aumentar la fidedignidad: Ayuda a que el modelo respete explícitamente las restricciones dadas por el usuario en el prompt en lugar de simplemente autocompletar texto
## ¿Cómo se entrena un Reward Model en el contexto de RLHF, cual es su función?

La función principal de un Reward Model (Modelo de Recompensa) es evaluar el texto generado por un modelo de lenguaje y asignarle una puntuación (score) que determine qué tan bueno es basándose estrictamente en las preferencias humanas. Su objetivo es actuar como el juez o evaluador automático durante la fase final de Aprendizaje por Refuerzo (RL), guiando al modelo principal para que genere respuestas alineadas con el usuario. Por lo general, este modelo evaluador suele ser un Gran Modelo de Lenguaje (LLM) de menor tamaño. El proceso para entrenar este modelo consta de los siguientes pasos:

1. Generación de respuestas: A partir de un conjunto de datos, se seleccionan diferentes instrucciones (prompts) y se procesan a través de un LLM (a veces variando la temperatura o configuración) para generar múltiples respuestas distintas para una misma petición.
2. Clasificación humana (Ranking): Evaluadores humanos leen estas múltiples respuestas y las ordenan (rankean) de mejor a peor según su criterio, o bien realizan una comparación binaria eligiendo cuál de las dos es la más adecuada.
3. Inicialización: Para comenzar su entrenamiento, el Modelo de Recompensa suele inicializarse utilizando los pesos de un modelo que ya fue ajustado previamente por instrucciones (Instruction Tuning).
4. Optimización matemática: Utilizando los rankings recopilados, se entrena al modelo de recompensa con una función de pérdida (loss) diseñada para que asigne sistemáticamente una puntuación más alta a las respuestas preferidas por los humanos y una más baja a las descartadas. Como detalle adicional, en sistemas avanzados como LLaMA 2, en lugar de un solo modelo se suelen entrenar múltiples Reward Models especializados: uno enfocado exclusivamente en calificar la utilidad (helpfulness) de la respuesta y otro dedicado a evaluar la seguridad y falta de toxicidad (safety) Explica RLHF, cual es el rol de PPO? Como se implementa? El RLHF (Aprendizaje por Refuerzo a partir de Retroalimentación Humana) formula el ajuste del modelo como un problema de aprendizaje por refuerzo clásico. En este esquema, el modelo de lenguaje representa la "política" (policy), las palabras del vocabulario son su "espacio de acción", y el Modelo de Recompensa evalúa los resultados de esa acción.
## ¿Cuál es el rol de PPO?

El algoritmo PPO (Optimización de Política Próxima o Proximal Policy Optimization) es el motor de optimización matemática en este proceso. Su rol es tomar la calificación que devolvió el Modelo de Recompensa y utilizarla para actualizar los pesos y parámetros internos del modelo de lenguaje, con el objetivo de maximizar esa recompensa futura.

## ¿Cómo se implementa en la práctica?

El flujo de implementación para actualizar el modelo sigue esta dinámica:

1. Generación: Se ingresa un prompt en el modelo principal (el cual ya fue inicializado y ajustado previamente mediante Supervised Fine-Tuning o SFT) y este genera un texto de respuesta.
2. Evaluación: El Modelo de Recompensa recibe esta respuesta y le asigna una puntuación escalar indicando qué tan alineada está con las preferencias humanas.
3. Actualización con PPO: El algoritmo PPO calcula cómo deben cambiar los parámetros del modelo generador para obtener puntajes más altos la próxima vez y aplica la actualización.
4. Restricción de Divergencia KL: Para evitar un problema grave llamado "sobreoptimización" (donde el modelo encuentra formas de engañar al Modelo de Recompensa perdiendo coherencia en el lenguaje), la implementación matemática incluye una penalización conocida como Divergencia KL. Esta restricción castiga al modelo si la distribución de sus nuevas predicciones se aleja demasiado de las predicciones originales del modelo SFT base
## ¿Qué es un reward model?

Un Modelo de Recompensa (Reward Model) es un sistema diseñado específicamente para capturar las preferencias humanas. Su función principal es analizar un texto generado y devolver un valor o puntuación (score) que determina qué tan bueno es ese texto en términos de lo que preferiría un humano. Esta puntuación sirve como guía para indicarle al modelo principal si está haciendo un buen trabajo o no. En la práctica, suele ser un Gran Modelo de Lenguaje (LLM) de menor tamaño. Por ejemplo, durante el desarrollo de GPT-3, OpenAI utilizó un modelo de 175 mil millones de parámetros y lo evaluó usando un Modelo de Recompensa mucho más pequeño, de 6 mil millones de parámetros.

## ¿Qué rol juega rejection sampling en LLama?

En LLaMA 2, el Rejection Sampling (muestreo por rechazo) juega un rol fundamental como una de las estrategias utilizadas durante la etapa de Aprendizaje por Refuerzo a partir de Retroalimentación Humana (RLHF), aplicándose junto con PPO. Su función es mejorar la política del modelo mediante un proceso iterativo de selección de las mejores respuestas para luego utilizarlas como nuevos datos de entrenamiento. El proceso funciona de la siguiente manera:

1.
2.
3. Generación de muestras: Para un prompt dado, se le pide al modelo que genere

varias respuestas distintas (K outputs). Clasificación (Ranking): Esas K respuestas son evaluadas y ordenadas por el Modelo de Recompensa (Reward Model). Ajuste Fino (SFT): Se elige la respuesta que obtuvo la puntuación más alta y se utiliza para hacer un nuevo ciclo de Ajuste Fino Supervisado (SFT) sobre el modelo. El uso del Rejection Sampling se basa en que existe una ganancia potencial significativa entre la recompensa mediana promedio que suele dar el modelo y la recompensa máxima que se puede obtener si se generan múltiples opciones. Durante el entrenamiento de la versión más grande de LLaMA 2 (70B), el equipo de Meta aplicó varias iteraciones de esta técnica para ir refinando el modelo paso a paso

## Que rol juega el CLS token en BERT?

El token [CLS] es un símbolo especial que se añade siempre al principio de cada ejemplo o secuencia de entrada en el modelo BERT. Su rol principal es actuar como una representación agregada de toda la secuencia para facilitar las tareas de clasificación. En la arquitectura del modelo, el estado oculto final correspondiente a este token específico se extrae y se utiliza directamente como entrada para las capas de decisión, permitiendo resolver tareas a nivel de secuencia completa, como predecir si una oración es la continuación de otra (NSP) durante el preentrenamiento, o para clasificar textos durante el fine-tuning.

## Qué es group query attention?

El Grouped Query Attention (GQA) o Atención de Consultas Agrupadas es una variante optimizada del mecanismo de atención que se utiliza en arquitecturas modernas de Grandes Modelos de Lenguaje, como LLaMA o Qwen. Su característica principal radica en que múltiples cabezas de consulta (Query heads) comparten las mismas cabezas de clave (Key) y valor (Value). A diferencia del mecanismo clásico de Multi-Head Attention (MHA), donde cada Query tiene asignada su propia matriz de Key y Value de forma individual, el GQA los agrupa. Por ejemplo, en algunas implementaciones el modelo puede tener 32 cabezas para Query pero únicamente 4 para Key y Value, lo que significa que un grupo de 8 cabezas de consulta compartirá un mismo par KV. El objetivo principal de esta arquitectura es reducir significativamente el uso de memoria y mejorar la eficiencia durante la inferencia. Al compartir las cabezas, se achica el tamaño del KV cache y se ahorran parámetros computacionales, manteniendo al mismo tiempo un rendimiento del modelo casi idéntico al de la atención tradicional

## Que es un Mixture of Experts, como lo implementa OSS?

Un Mixture of Experts (MoE) o Mezcla de Expertos es una arquitectura que permite aumentar significativamente la capacidad y calidad de un modelo sin incrementar proporcionalmente su costo computacional. Funciona reemplazando las capas densas tradicionales (las redes Feed-Forward) por capas dispersas (sparsas) compuestas por múltiples sub-redes independientes llamadas "expertos". Además, incluye un mecanismo de enrutamiento (router o gate) que decide a qué expertos específicos enviar cada palabra o token. En lugar de usar toda la red, el router envía el token solo a los mejores expertos (top-k) para esa tarea particular. Así, el modelo posee una cantidad gigante de parámetros en total, pero para cada cálculo utiliza solo una pequeña fracción de ellos, procesando la información a la misma velocidad que un modelo más chico.

## ¿Cómo lo implementa GPT-OSS?

Según los datos del modelo GPT-OSS-20B, su implementación de MoE funciona de la siguiente manera:

- Cuenta con un total de 32 expertos.
- El router está configurado para activar únicamente a los 4 mejores expertos (top-4) por cada token.
- Como resultado, aunque el modelo tiene 21.000 millones de parámetros totales, solo utiliza 3.600 millones de parámetros activos al procesar la información, logrando gran eficiencia
## ¿Qué es el KV cache?

¿Por qué es necesario durante la generación autoregresiva y cómo impacta en la memoria? El KV cache (Caché de Claves y Valores) es una técnica de optimización esencial utilizada en la arquitectura Transformer durante la fase de inferencia. (Aclaración: El detalle técnico de este concepto no se explica en las diapositivas proporcionadas, donde solo aparece listado en las preguntas de la guía. Por ello, complemento la respuesta con información externa para que puedas resolver tu duda).

## ¿Por qué es necesario durante la generación autoregresiva?

En los modelos autoregresivos, la generación de texto se realiza de forma secuencial, "palabra por palabra". Para predecir un nuevo token, el mecanismo de atención del modelo necesita mirar hacia atrás y relacionar la consulta (Query) actual con las claves (Keys) y valores (Values) de todos los tokens anteriores. Sin un caché, el modelo tendría que recalcular matemáticamente las matrices K y V de toda la historia de la secuencia en cada uno de los pasos. El KV cache soluciona esto almacenando en la memoria estos vectores ya calculados. De esta forma, para generar una nueva palabra, el modelo solo procesa el K y V del nuevo token y reutiliza el caché para el resto, acelerando drásticamente la velocidad de inferencia y evitando cálculos redundantes.

## ¿Cómo impacta en la memoria?

El impacto en la memoria de video (VRAM de la GPU) es altísimo y representa el mayor cuello de botella operativo de los Grandes Modelos de Lenguaje. Sus principales características de consumo son:

- Crecimiento dinámico: El KV cache crece linealmente a medida que la secuencia se hace más larga. Si se duplica la longitud del contexto (por ejemplo, de 4.000 a 8.000 tokens), el tamaño del caché también se duplica.
- Impacto por usuario: Si estás sirviendo a múltiples usuarios concurrentes, el sistema debe almacenar un KV cache único y separado para cada sesión y petición.

Es justamente debido a este masivo consumo de memoria que las arquitecturas modernas implementan optimizaciones como el Grouped Query Attention (GQA) —del cual hablamos anteriormente—, que agrupa y comparte las cabezas de Key y Value para reducir significativamente el tamaño del caché que se debe almacenar en la GPU

## ¿Por qué tokenizar a nivel de caracteres genera más costo computacional que usar subwords como BPE?

¿Qué relación tiene con la cantidad de pasos de inferencia? Tokenizar a nivel de caracteres genera un mayor costo computacional frente a usar subwords (como BPE) debido al impacto directo que tiene en la longitud final de la secuencia de texto. La relación de esto con la cantidad de pasos de inferencia y la memoria se explica por los siguientes factores:

- Más pasos de inferencia (generación autoregresiva): Los modelos generan texto secuencialmente, token por token. Al usar caracteres, una misma palabra requiere muchos más tokens para formarse y, en consecuencia, más tiempo de procesamiento.

Por ejemplo, la palabra "lowest" podría procesarse en solo dos tokens con BPE ("low", "est"), requiriendo apenas dos pasos de inferencia; sin embargo, requeriría seis tokens individuales (y seis pasos de cálculo) si se usan caracteres.

- Crecimiento cuadrático de la atención: La complejidad computacional del mecanismo de self-attention crece de forma cuadrática () respecto al largo de la secuencia. Una secuencia fragmentada en decenas de caracteres en lugar de pocas subwords multiplica exponencialmente la cantidad de operaciones matemáticas que el modelo debe realizar.
- Saturación del KV cache: Como mencionamos en la pregunta anterior, el consumo de memoria del KV cache crece linealmente a medida que la secuencia se alarga.

Procesar oraciones a nivel de caracteres genera historiales enormes, llenando rápidamente la memoria VRAM de la GPU y limitando la eficiencia del sistema

## ¿Cuál es la complejidad computacional de self-attention con respecto al largo de la secuencia?

¿Qué implicancias prácticas tiene? La complejidad computacional del mecanismo de self-attention crece de forma cuadrática () con respecto al largo de la secuencia (). Esto se debe a que, en la arquitectura Transformer, el cálculo de atención requiere que cada token (actuando como Query) realice un producto escalar contra todos y cada uno de los demás tokens de la secuencia (actuando como Keys) para obtener sus puntuaciones de atención. Esta naturaleza cuadrática trae consigo varias implicancias prácticas fundamentales en el diseño y uso de los modelos:

- Problemas de escalabilidad: A medida que se incrementa la longitud de la secuencia, la cantidad de operaciones matemáticas se multiplica de forma exponencial. Esto introduce graves cuellos de botella computacionales y hace que procesar textos o secuencias demasiado grandes se vuelva inmanejable.
- Límites estrictos en la ventana de contexto: El enorme crecimiento en la demanda de cómputo y memoria es el factor técnico principal que determina por qué todos los modelos tienen un límite máximo fijo en su ventana de contexto.
- Impacto drástico en la memoria: Si decidís duplicar el largo del contexto de un modelo (por ejemplo, pasando de 4.000 a 8.000 tokens), el costo computacional de la atención se cuadruplica y el uso de la memoria de la GPU (especialmente por el almacenamiento del KV cache) se incrementa masivamente.
## ¿Por qué se necesita la feed-forward network (FFN) en cada bloque del Transformer además de self-attention?

¿Qué aporta? Para responder a esta pregunta en detalle y no ser sintético, es importante entender primero cómo se divide el trabajo dentro de un bloque de la arquitectura Transformer. Dentro de cada capa o bloque del modelo (ya sea en el encoder o en el decoder), existen dos subcapas principales que trabajan en equipo: la Atención Multi-Cabeza (Multi-Head Attention) y la Red Feed-Forward (FFN). Cada una tiene un rol específico y complementario:

1. El rol de Self-Attention (El "Enrutador" de Contexto): El mecanismo de atención se encarga exclusivamente de la relación entre los tokens. Su trabajo es mirar toda la secuencia y calcular puntuaciones (scores) para determinar cuánto peso o "atención" debe prestarle un token específico al resto de las palabras de la oración. Como resultado de este cruce de información, produce un vector de contexto. Sin embargo, la atención por sí sola es mayormente una operación de promedios ponderados y combinaciones lineales.
2. El rol de la Feed-Forward Network (El "Refinador"): Una vez que el mecanismo de atención recolectó la información del contexto, ese vector resultante se utiliza como entrada para la red Feed-Forward, la cual se encarga de generar la salida final de esa subcapa. Su función principal es actualizar y refinar el estado oculto (hidden state) del token antes de pasarlo a la siguiente capa del modelo. (Aclaración: Las diapositivas de las fuentes establecen esta estructura general, pero para explicarte el "por qué" técnico exacto y matemático de su necesidad, incorporo a continuación información externa a tus documentos). La inclusión de la FFN aporta tres elementos críticos que la atención por sí sola no puede resolver:
- Transformación No Lineal: El mecanismo de self-attention consiste en multiplicaciones de matrices y sumas ponderadas, lo cual es matemáticamente lineal.

Si apiláramos capas de atención pura, el modelo colapsaría en una única transformación lineal gigante, limitando severamente su inteligencia. La FFN introduce funciones de activación no lineales (como la función gelu, mencionada en los hiperparámetros de BERT en tus fuentes), permitiendo que el modelo aprenda patrones altamente complejos.

- Procesamiento Individual por Token: Mientras que la capa de atención mezcla información entre múltiples tokens diferentes, la FFN se aplica posición por posición. Toma la nueva representación contextualizada de un token específico y la procesa de forma aislada para decidir qué características de ese contexto son realmente útiles.
- Capacidad de Memoria y Parámetros: Tal como insinúa la pregunta 129 de tu guía de estudio, las FFN suelen contener muchísimos más parámetros que las capas de atención. A nivel arquitectónico, la FFN primero expande la información a una dimensión mucho mayor (conocido como intermediate size) y luego la vuelve a comprimir. Este enorme espacio de parámetros funciona en la práctica como una gigantesca "memoria de hechos" o un banco de conocimiento donde el modelo almacena lo que sabe sobre el mundo y el lenguaje.

En resumen, la capa de self-attention decide "hacia dónde mirar" para recopilar el contexto adecuado, y la capa FFN decide "qué pensar" sobre esa información, transformándola y procesándola en profundidad antes de avanzar al siguiente nivel de abstracción.

## ¿Qué es teacher forcing?

¿Qué es exposure bias y cómo se mitiga? El Teacher Forcing (forzado del profesor) es una técnica utilizada durante el entrenamiento de modelos generativos, como las Redes Neuronales Recurrentes (RNN). Consiste en alimentar al modelo en cada paso de tiempo con el token real o correcto (ground truth, ) extraído de los datos de entrenamiento para que intente predecir la siguiente palabra, en lugar de utilizar la predicción que el propio modelo acaba de generar en el paso anterior (). Su principal ventaja es que acelera drásticamente la convergencia y estabiliza el entrenamiento, ya que evita que el modelo se desvíe por caminos erróneos al inicio de su aprendizaje. El Exposure Bias (sesgo de exposición) es una discrepancia o problema fundamental que surge como consecuencia directa de haber utilizado Teacher Forcing. Este fenómeno ocurre porque durante la fase de inferencia (es decir, cuando el usuario ya está interactuando con el modelo y este debe generar texto nuevo), no existe un ground truth externo que lo guíe. En este escenario, el modelo está obligado a consumir sus propias predicciones pasadas como entrada para generar el siguiente paso. Al estar acostumbrado a recibir siempre la palabra perfecta durante el entrenamiento, si el modelo comete un error, no sabe cómo corregirlo, provocando que el error se propague y se amplifique a lo largo de toda la secuencia. Las fuentes destacan que este sesgo de exposición es una de las influencias del entrenamiento que contribuyen a que los modelos generen alucinaciones.

## ¿Cómo se mitiga?

Las fuentes proporcionadas no detallan una técnica algorítmica específica e individual (como el muestreo programado o Scheduled Sampling, que es información externa) para solucionar exclusivamente el sesgo de exposición. Sin embargo, abordan su mitigación dentro de las estrategias generales para reducir las alucinaciones derivadas del proceso de entrenamiento. Para mitigarlo, se recomienda:

- Mejorar los modelos: Intervenir y corregir estas vulnerabilidades de decodificación que ocurren durante el entrenamiento.
- Aprendizaje por refuerzo (RLHF): Utilizar modelos de recompensa (Reward models) para penalizar estas desviaciones y alinear el texto generado de modo que el modelo aprenda a recuperarse de sus propios errores sin propagar información factual errónea
## ¿Qué son las scaling laws?

¿Qué cambió con Chinchilla respecto a la receta de entrenamiento? Las leyes de escalado (scaling laws) son principios empíricos que describen cómo el rendimiento de un modelo de lenguaje (evaluado a través de la reducción de su error o pérdida) mejora de manera predecible y log-lineal a medida que se incrementan tres variables fundamentales: la cantidad de parámetros del modelo (N), el tamaño del conjunto de datos (D) y el volumen de cómputo (C) empleado durante el entrenamiento. En esencia, dictan que los modelos seguirán mejorando sistemáticamente siempre y cuando cuenten con el poder computacional y los datos suficientes.

## ¿Qué cambió con Chinchilla respecto a la receta de entrenamiento?

La investigación detrás del modelo Chinchilla (desarrollado en 2022) modificó drásticamente la estrategia óptima de entrenamiento, redefiniendo cómo se debían escalar los modelos. Sus principales aportes fueron:

- Evidenció que los modelos estaban subentrenados: Antes de Chinchilla, la tendencia de la industria (con modelos como GPT-3 o Gopher) era incrementar masivamente el número de parámetros (alcanzando 175 o 280 mil millones) pero manteniendo la cantidad de datos relativamente baja (alrededor de 300 mil millones de tokens). Chinchilla demostró que este enfoque era ineficiente.
- Priorizar datos sobre parámetros: La nueva receta de entrenamiento establece que se debe escalar el modelo y el dataset proporcionalmente para obtener un rendimiento óptimo. Es decir, en lugar de simplemente hacer la red neuronal más grande, es fundamental priorizar la inyección de más datos de entrenamiento.
- La prueba en la práctica: Para validarlo, entrenaron Chinchilla con "solo" 70 mil millones de parámetros (mucho más pequeño que sus predecesores), pero lo alimentaron con una cantidad masiva de 1.4 billones (trillions) de tokens. Esto demostró que un modelo más chico, pero entrenado con los datos adecuados, podía superar a modelos gigantes.
- Mejor planificación de recursos: Dado que el costo de inferencia y entrenamiento crece de forma superlineal con el tamaño del modelo (parámetros), esta nueva estrategia de escalar balanceando datos y cómputo permite optimizar los costos de manera mucho más efectiva
## ¿Qué es DPO (Direct Preference Optimization)?

¿Cómo simplifica el pipeline de RLHF? El DPO (Direct Preference Optimization) es una técnica de entrenamiento diseñada para alinear modelos de lenguaje con las preferencias humanas, con la particularidad de que no necesita entrenar un Modelo de Recompensa (Reward Model). En lugar de optimizar en función de una puntuación de recompensa externa, DPO plantea el entrenamiento evaluando directamente las desviaciones relativas entre un ejemplo "bueno" (elegido) y uno "malo" (rechazado) respecto al modelo original.

## ¿Cómo simplifica el pipeline de RLHF?

El pipeline tradicional de RLHF es inestable y muy complejo, ya que exige ejecutar múltiples pasos y mantener en la memoria varios modelos distintos al mismo tiempo: un pequeño Modelo de Recompensa, el LLM que se está ajustando y un modelo congelado de referencia. DPO simplifica drásticamente este flujo al eliminar por completo el Modelo de Recompensa. Al no necesitar este modelo extra para calcular la función de pérdida (loss), DPO logra el mismo objetivo matemático de maximizar las preferencias con restricción de divergencia KL, pero en un solo paso directo, haciendo que la optimización sea mucho más eficiente

## Describí la "receta moderna" de un Transformer (LLaMA/Mistral): ¿qué

cambió respecto al Transformer original de 2017 (RMSNorm, SwiGLU, RoPE, GQA)? La "receta moderna" de arquitecturas como LLaMA o Mistral (que hoy en día se consolidaron exclusivamente como modelos Decoder-only) introduce optimizaciones críticas respecto al Transformer original de 2017 enfocadas en mejorar la eficiencia, el consumo de memoria y la estabilidad en el escalado:

- Normalización (Pre-Norm y RMSNorm): Se abandona el LayerNorm clásico en favor del RMSNorm, que es computacionalmente más barato y ofrece una estabilidad similar. Además, la normalización ahora se aplica antes de entrar a la capa de atención o FFN (Pre-Norm en lugar de Post-Norm), lo cual estabiliza enormemente el entrenamiento de modelos de más de 100 capas sin requerir warmup de la tasa de aprendizaje.
- Activación FFN (SwiGLU): Las clásicas funciones de activación (como ReLU o

GELU) dentro de la subcapa de Feed-Forward se reemplazaron por SwiGLU. Esta variante introduce un mecanismo de compuerta multiplicativo (gating) que mejora el rendimiento general del modelo utilizando menos parámetros efectivos.

- Codificación Posicional (RoPE): Se dejó de sumar una posición absoluta estática a la entrada. En su lugar, se utilizan Rotary Position Embeddings (RoPE), un método que aplica una rotación matemática directamente sobre las matrices Query y Key en cada paso de atención. Esto permite que el modelo entienda nativamente la "distancia relativa" entre palabras y generalice de forma excelente secuencias de texto mucho más largas.
- Eficiencia de Atención (GQA): Para resolver el embotellamiento de memoria durante la generación de texto, el Multi-Head Attention original se sustituyó por

Grouped Query Attention (GQA). Al hacer que múltiples cabezas de búsqueda (Query) compartan el mismo par de Key/Value, se logra achicar drásticamente el tamaño del KV cache que debe alojarse en la GPU, manteniendo una calidad de respuesta casi idéntica

## ¿Cómo conectás KV cache → GQA → eficiencia de inferencia?

¿Por qué GQA reduce el uso de memoria? La conexión entre estos tres conceptos radica en cómo se optimiza el manejo de la memoria durante la generación de texto paso a paso. Se pueden conectar de la siguiente manera:

1. El problema del KV cache (Cuello de botella) Como conversamos anteriormente, durante la inferencia autoregresiva el modelo genera texto palabra por palabra. Para no recalcular matemáticamente todo el contexto en cada paso, almacena los vectores Key y Value de los tokens anteriores en la memoria de la GPU (KV cache). El problema es que este caché es gigantesco, crece linealmente con el tamaño del texto y rápidamente agota la memoria VRAM, convirtiéndose en el principal cuello de botella.
2. La intervención de GQA (Grouped Query Attention) Para solucionar este problema de memoria, se introduce la arquitectura GQA. En el mecanismo de atención tradicional (Multi- head Attention o MHA), cada cabeza de Query (Q) tiene asignada su propia cabeza individual de Key (K) y Value (V). En cambio, GQA agrupa múltiples cabezas de Query para que compartan la misma cabeza de Key y Value. Por ejemplo, en modelos como GPT-OSS, se utilizan 32 cabezas de Query pero únicamente 4 cabezas de Key/Value compartidas.
3. El resultado: Reducción de memoria y Eficiencia de Inferencia GQA reduce el uso de memoria precisamente por este mecanismo de "compartir". Al reducir drásticamente la cantidad total de cabezas K y V que existen en la arquitectura, el tamaño del KV cache que debe almacenarse en la memoria se achica proporcionalmente. Esta reducción de memoria impacta directamente en la eficiencia de inferencia:
- Al haber menos datos pesados que mover dentro de la memoria de la GPU, los cálculos matemáticos para generar cada token se vuelven más rápidos.
- Al liberar tanto espacio de VRAM, el sistema gana la capacidad de procesar secuencias de texto (ventanas de contexto) muchísimo más largas, o bien, permite servir a una mayor cantidad de usuarios concurrentes utilizando exactamente el mismo hardware.
## ¿Por qué es necesario dividir por √dₖ en la puntuación de atención?

¿Qué pasa si no se escala? Es necesario dividir por la raíz cuadrada de la dimensión de las claves (√dₖ) para normalizar la varianza resultante del producto escalar entre la consulta (Query, ) y la clave (Key, ). Matemáticamente, si los vectores y tienen componentes con media 0 y varianza 1, su producto escalar tendrá una varianza igual a . Esto significa que, en arquitecturas donde la dimensión es muy grande, los valores obtenidos al multiplicar crecerán de forma desproporcionada. Si no se escala dividiendo por √dₖ, estos valores tan altos empujarán a la función softmax hacia regiones de saturación. En estas regiones planas de la función, los gradientes se vuelven extremadamente pequeños (prácticamente nulos), lo que provoca que el modelo deje de aprender por completo. Al aplicar la división por √dₖ, se contrarresta este efecto y se garantiza que los valores ingresen a la función softmax en una zona de gradientes saludables e informativos, permitiendo que el entrenamiento avance correctamente

## ¿Cómo funciona el causal masking en la masked self-attention del decoder?

El causal masking (o enmascaramiento causal) se utiliza en la subcapa de Masked Self- Attention del decodificador (decoder) para garantizar que el modelo genere texto de manera secuencial, impidiendo que haga trampa al "ver el futuro" de la secuencia.

## ¿Cómo funciona en la práctica?

En los modelos autoregresivos, cada nuevo token que se genera solo puede (y debe) prestarle atención a los tokens que vinieron antes que él. Para forzar esta regla matemática, se aplica una máscara triangular sobre las puntuaciones de atención antes de calcular los pesos finales. El proceso es el siguiente:

1. Cálculo normal para el pasado: Cuando el modelo calcula la atención para una palabra en la posición i, realiza el cálculo de similitud normal (el producto escalar dividido por √dₖ) contra todas las palabras anteriores y contra sí misma.
2. Enmascaramiento del futuro: Para cualquier palabra que se encuentre en una posición futura (es decir, a la derecha de la palabra actual), el mecanismo interviene y fuerza su puntuación de atención a infinito negativo ().
3. El efecto de la función Softmax: Una vez aplicada esta máscara, todos los valores pasan por la función softmax para convertirse en probabilidades o "pesos". Matemáticamente, el softmax aplicado a da como resultado exactamente 0. Como consecuencia directa, los pesos de atención para todos los tokens futuros se anulan por completo. Esto destruye cualquier conexión con las palabras que aún no se "leyeron", obligando al decoder a apoyarse única y exclusivamente en el contexto que ya generó en el pasado.
## ¿Por qué un vocabulario más chico (o tokenización a nivel de caracteres) aumenta el costo de inferencia?

¿Qué relación tiene con la cantidad de pasos autoregresivos y el KV cache? Un vocabulario más chico, como en el caso de la tokenización a nivel de caracteres, aumenta el costo de inferencia porque obliga al modelo a fragmentar la información y trabajar con secuencias de tokens mucho más largas. Es decir, lo que en un vocabulario amplio se representaría con una sola palabra o pieza, aquí se divide en múltiples tokens individuales. La relación de esto con el costo computacional se explica por dos grandes factores:

- Impacto en los pasos autoregresivos: En las arquitecturas decoder-only, la generación de texto se realiza de manera autoregresiva, lo que significa que el modelo procesa y predice la salida paso a paso, un token a la vez. Al tokenizar por caracteres, una palabra corta requiere múltiples pasos de cálculo en lugar de uno solo. Por lo tanto, el modelo debe ejecutar muchas más iteraciones para generar la misma frase, lo que eleva el costo de cómputo y el tiempo de inferencia.
- Saturación del KV cache: Para no recalcular toda la oración en cada uno de estos pasos, el modelo almacena en memoria los vectores Key y Value de los tokens pasados (el KV cache). Al producir secuencias excesivamente largas por culpa de un vocabulario reducido, este caché crece a un ritmo mucho más acelerado. Dado que procesar grandes secuencias de contexto se vuelve computacionalmente inmanejable debido a la naturaleza cuadrática de la atención, este crecimiento rápido agota la memoria VRAM y reduce la eficiencia general del sistema.

Si una oración se tokeniza en 10 tokens con BPE y en 45 caracteres sin BPE,

## ¿cómo cambia el costo de inferencia y el uso de memoria?

Pensalo en términos de pasos autoregresivos, tamaño de KV cache y complejidad O(n²) de atención. Tokenizar la oración en 45 caracteres en lugar de 10 tokens BPE incrementa dramáticamente tanto el tiempo de procesamiento como el consumo de memoria. Analizado desde los tres factores que mencionás, el impacto es el siguiente:

- Pasos autoregresivos: En las arquitecturas decoder-only, la generación se realiza de manera autoregresiva, prediciendo estrictamente un token a la vez. Por lo tanto, generar la oración en caracteres requerirá 45 iteraciones o pasos de cálculo secuenciales en lugar de solo 10. Esto hace que el proceso de generación sea 4.5 veces más lento y requiera invocar el modelo muchas más veces para obtener el mismo resultado de texto.
- Tamaño del KV cache: Como el modelo debe almacenar en la VRAM los vectores de Key y Value por cada token procesado para evitar recalcularlos, una secuencia fragmentada en 45 tokens ocupará 4.5 veces más memoria en el KV cache que su contraparte de 10 tokens BPE.
- Complejidad O(n²) de atención: El mecanismo de self-attention tiene un crecimiento cuadrático con respecto al largo de la secuencia (). o Para 10 tokens BPE, la complejidad proporcional es de . o Para 45 caracteres, la complejidad se dispara a . o Esto significa que el modelo necesita realizar más de 20 veces más operaciones matemáticas cruzadas en sus capas de atención para entender el contexto de la oración letra por letra, comparado con hacerlo utilizando fragmentos de palabras (subwords).

En conclusión, la tokenización clásica por caracteres presenta la gran desventaja de producir secuencias muy largas, lo que satura rápidamente la memoria y multiplica exponencialmente el costo computacional de la inferencia debido a la naturaleza de la arquitectura Transformer.

## ¿Por qué las APIs de LLMs cobran por token y no por palabra o por request?

¿Qué costos subyacentes refleja esa decisión? Las APIs de los Grandes Modelos de Lenguaje (LLMs) cobran por token y no por palabra o por petición (request) porque el token es la unidad mínima y fundamental con la que el modelo procesa la información. (Aclaración: Si bien tu guía de estudio plantea esta interrogante en la pregunta 120, las fuentes documentales no ofrecen una respuesta comercial explícita sobre las APIs. Sin embargo, basándome en los fundamentos técnicos de la arquitectura Transformer presentes en tus documentos y en nuestra conversación anterior, a continuación te explico cómo se justifica esta decisión). Cobrar por token es la única métrica que refleja con exactitud el gasto real de los recursos computacionales subyacentes, por las siguientes razones:

## ¿Por qué no cobrar por palabra?

Los modelos modernos no leen palabras completas, sino que utilizan algoritmos de tokenización de sub-palabras (subwords) como BPE o WordPiece. Esto significa que una palabra puede dividirse en múltiples tokens dependiendo de su rareza o de su idioma (por ejemplo, la palabra "lowest" puede segmentarse en "low" y "est"). Como el modelo genera el texto de forma secuencial y autoregresiva (un token a la vez), cada uno de estos fragmentos exige una iteración completa de cálculo matemático. Cobrar por "palabra" sería ciego al verdadero esfuerzo del modelo, ya que palabras complejas consumen mucho más cómputo que palabras simples.

## ¿Por qué no cobrar por petición (request)?

El largo de una petición puede variar enormemente desde una simple oración hasta un documento de miles de palabras. Como hemos visto, la arquitectura Transformer tiene dos grandes limitantes técnicos:

1. Crecimiento cuadrático de la atención: La complejidad de los cálculos matemáticos del mecanismo de self-attention crece de forma cuadrática () respecto a la cantidad de tokens.
2. Saturación del KV cache: El modelo debe reservar un espacio físico en la memoria de video (VRAM) por cada token que lee o genera. Una petición larga consume exponencialmente más memoria y poder de procesamiento que una corta, por lo que cobrar una tarifa plana por request llevaría a la pérdida económica para el proveedor o a un cobro injusto para el usuario con peticiones cortas.
## ¿Qué costos subyacentes refleja esta decisión?

Cobrar estrictamente por el volumen de tokens procesados refleja directamente los significativos costos de hosting, infraestructura e inferencia. Los sistemas de Inteligencia Artificial requieren un hardware masivo y costoso impulsado por una creciente demanda de chips GPU de altísimo consumo energético. En definitiva, la facturación por token traduce al precio el uso exacto de:

- Poder de cálculo (FLOPs): La cantidad de operaciones matriciales necesarias para atravesar los miles de millones de parámetros en las subcapas de atención y redes

Feed-Forward.

- Alquiler de memoria VRAM: El espacio vital y limitado de la GPU que tu petición está acaparando al alojar los historiales en el KV cache
## ¿Qué diferencia hay entre tokens de input (prefill) y tokens de output (decode) en términos de cómputo?

¿Por qué algunas APIs cobran distinto por cada uno? La diferencia fundamental en términos de cómputo entre estos dos tipos de tokens radica en la paralelización frente a la secuencialidad inherente a la arquitectura Transformer. Tokens de Input (Fase de Prefill): Cuando le enviás un texto o prompt al modelo, este no lo lee palabra por palabra. La arquitectura Transformer tiene la capacidad de procesar todos los elementos de la secuencia de entrada simultáneamente en paralelo. Durante esta fase (conocida como prefill), el modelo calcula las relaciones entre todas las palabras de tu pregunta de un solo golpe y genera el KV cache inicial. Al aprovechar el procesamiento paralelo masivo de las GPUs, el costo de tiempo y cómputo por token es sumamente bajo y eficiente. Tokens de Output (Fase de Decode): Por el contrario, la generación de la respuesta es un proceso estrictamente autoregresivo, lo que significa que el modelo está obligado a generar la salida paso a paso, prediciendo un token a la vez basándose en el contexto previo. Al no poder "ver el futuro", el modelo no puede paralelizar este proceso. Para generar cada nueva palabra, debe realizar una iteración completa consultando el KV cache (el historial). Esta naturaleza secuencial hace que generar tokens de salida sea el mayor cuello de botella del sistema y resulte computacionalmente mucho más pesado y lento.

## ¿Por qué las APIs cobran distinto por cada uno?

(Aclaración: Los detalles comerciales específicos de las APIs no están explícitos en tus diapositivas, pero la decisión de facturación es una consecuencia directa de las limitaciones de hardware explicadas arriba, por lo que integro información externa para completarte el panorama). Las empresas proveedoras de LLMs suelen cobrar los tokens de input mucho más baratos que los de output porque los precios reflejan directamente la eficiencia con la que se utiliza el hardware subyacente:

1. Eficiencia de hardware (Prefill): Procesar el input satura eficientemente los núcleos de cálculo de la GPU, realizando operaciones matemáticas gigantes en una sola ráfaga rápida. Cuesta poco tiempo de servidor.
2. Cuello de botella de memoria (Decode): Generar el output infrautiliza la capacidad de cálculo puro de la GPU porque el proceso debe esperar a que el token anterior termine de generarse. En esta fase, el tiempo total está determinado por la velocidad a la que la memoria VRAM puede transferir el enorme KV cache en cada uno de los pasos iterativos. Al requerir mucho más tiempo exclusivo ("alquiler") de la GPU para generar una respuesta que para leer una pregunta de la misma longitud, el costo operativo del proveedor se dispara en el decode, trasladando esa diferencia al precio final de la API.
## ¿Qué es el prefill y qué es el decode en la inferencia de un LLM?

¿Por qué el prefill se puede paralelizar pero el decode no? Esta pregunta está muy ligada a los conceptos técnicos de la pregunta anterior, pero enfocada específicamente en las fases de funcionamiento del modelo durante la inferencia. En la inferencia de un Gran Modelo de Lenguaje (LLM), el trabajo se divide en dos fases bien marcadas:

1. La fase de Prefill (Pre-llenado): Es la etapa inicial donde el modelo recibe la secuencia de entrada (tu prompt o pregunta) y la procesa para entender su contexto. Su objetivo técnico principal es calcular las representaciones matemáticas iniciales y llenar el KV cache con la información de toda tu pregunta antes de empezar a responder.
2. La fase de Decode (Decodificación): Es la etapa de generación de texto propiamente dicha. Una vez que el modelo entendió la entrada, comienza a producir la respuesta, escupiendo los nuevos tokens hacia el usuario.
## ¿Por qué el prefill se puede paralelizar pero el decode no?

La diferencia radica en cómo la arquitectura maneja la información conocida versus la desconocida:

- El prefill se puede paralelizar porque, al momento de enviar el prompt, todas las palabras de la entrada ya existen y están disponibles. Una de las grandes ventajas de la arquitectura Transformer frente a modelos más antiguos (como las RNN) es que puede procesar todos los elementos de una secuencia simultáneamente. Al conocer todo el texto de antemano, el modelo utiliza los miles de núcleos de la GPU para calcular las relaciones matemáticas de todas las palabras al mismo tiempo, en un solo bloque rápido y masivo.
- El decode no se puede paralelizar porque la generación de texto es un proceso por diseño autorregresivo, es decir, el modelo predice la próxima palabra basándose estrictamente en los tokens anteriores. Como el modelo no puede "ver el futuro", es matemáticamente imposible que calcule la quinta palabra de su respuesta si todavía no generó la cuarta. Esto lo obliga a procesar la salida de forma secuencial y paso a paso, convirtiendo la fase de decode en el proceso más lento y costoso de la inferencia.
## ¿Cómo crece el costo computacional de generar una secuencia de largo L?

## ¿Qué rol juegan la complejidad O(n²) de atención y el KV cache en ese costo?

El costo computacional de generar una secuencia de largo no es constante, sino que crece de forma cuadrática () y es altamente dependiente de la memoria. Como la generación de texto en estos modelos se realiza estrictamente paso a paso ("palabra por palabra"), cada nuevo token generado añade una carga adicional al sistema. En este proceso, los dos factores técnicos que definen el costo son la complejidad de la atención y el uso del KV cache:

1. El rol de la complejidad cuadrática de atención: En la arquitectura Transformer, cada vez que el modelo intenta predecir el siguiente token de la secuencia, su mecanismo de atención requiere relacionar matemáticamente su Query actual contra los Keys de todos y cada uno de los tokens anteriores. Por lo tanto, el esfuerzo de cálculo no es igual al principio de la oración que al final. A medida que crece, la cantidad de operaciones matemáticas de similitud que el modelo debe procesar se multiplica exponencialmente. Esta naturaleza cuadrática es el principal factor que encarece el cómputo y hace inmanejable procesar secuencias de contexto infinitas.
2. El rol del KV cache (Caché de Claves y Valores): Para que el costo de tiempo computacional no sea infinito (es decir, para evitar tener que recalcular las matrices pasadas en cada uno de los pasos), el modelo guarda en la memoria VRAM el historial de cálculos mediante el KV cache. Si bien esto acelera enormemente la velocidad de inferencia, introduce un nuevo problema: el crecimiento lineal del consumo de memoria. A medida que la secuencia de largo se incrementa, el tamaño de este caché crece linealmente, ocupando cada vez más espacio físico en la GPU. En secuencias muy largas, el cuello de botella computacional deja de ser la capacidad de hacer cálculos matemáticos (los FLOPs) y pasa a ser la lentitud de la memoria para transferir este gigantesco KV cache en cada uno de los pasos autoregresivos. En conclusión, generar una secuencia de largo enfrenta un crecimiento cuadrático en la cantidad de operaciones de atención que debe realizar, y un crecimiento lineal agresivo en la memoria que necesita consumir para evitar colapsar operativamente. Si duplicás el largo de contexto de un modelo (e.g., de 4K a 8K tokens),
## ¿qué pasa con el uso de memoria del KV cache y con el costo de atención?

Si duplicás el largo del contexto de un modelo, el impacto es el siguiente:

1. Uso de memoria del KV cache: Se duplica. Como el sistema necesita guardar en la VRAM los vectores clave y valor de cada token individual, el tamaño de este caché crece de forma estrictamente lineal respecto al largo de la secuencia.
2. Costo de atención (Cómputo): Se cuadruplica. Debido a que la complejidad computacional del mecanismo de self-attention es cuadrática (O(N2)), al multiplicar la longitud de la secuencia por 2, la cantidad de operaciones matemáticas necesarias se multiplica por 4 (22).
## ¿Por qué modelos como LLaMA dividen los números en dígitos individuales durante la tokenización?

¿Qué trade-off implica esa decisión? Modelos como LLaMA (que utiliza el algoritmo BPE con un vocabulario de 32.000 tokens) deciden dividir los números en dígitos individuales principalmente para lograr una mejor cobertura de los datos. Dado que los números pueden tener combinaciones infinitas, intentar almacenar números enteros o grandes fragmentos numéricos saturaría rápidamente el vocabulario y generaría el riesgo de dejar información fuera de este (el problema de palabras raras o Out-Of- Vocabulary). Al dividirlos en dígitos únicos (del 0 al 9), el modelo se garantiza matemáticamente la capacidad de representar cualquier número que exista sin desperdiciar espacio valioso en su diccionario interno. El trade-off (costo-beneficio) de esta decisión: El gran precio a pagar por esta cobertura perfecta es que se generan secuencias de tokens más largas. Tal como conversamos antes respecto a la tokenización a nivel de caracteres, esta fragmentación arrastra consecuencias directas en la eficiencia de inferencia:

- Más pasos autoregresivos: Para generar un número grande (por ejemplo, "1.500.000"), el decodificador deberá realizar al menos 7 iteraciones o pasos de cálculo secuenciales, prediciendo estrictamente dígito por dígito.
- Aumento en la memoria y cómputo: Cada uno de esos dígitos individuales requerirá su propio espacio de almacenamiento en el KV cache y sumará carga a la complejidad cuadrática (O(N2)) del mecanismo de atención, haciendo que el procesamiento sea marginalmente más lento y pesado.
## ¿Por qué la ventana de contexto tiene un límite máximo?

¿Qué factores técnicos (memoria, cómputo, positional encoding) lo determinan? La ventana de contexto de un modelo de lenguaje tiene un límite máximo debido a restricciones físicas y matemáticas intrínsecas a la arquitectura Transformer, las cuales imponen severos cuellos de botella a medida que las secuencias de texto se hacen más largas. Los tres factores técnicos principales que determinan e imponen este límite son:

1. Cómputo (Complejidad Cuadrática de la Atención): Como hemos conversado anteriormente, el mecanismo de self-attention tiene una complejidad computacional que crece de forma cuadrática (O(N2)) con respecto a la longitud de la secuencia (N). Para comprender el contexto de manera global, el modelo exige que cada token realice cálculos matemáticos (producto escalar) contra todos y cada uno de los tokens anteriores en la oración. En consecuencia, si se duplica la ventana de contexto, la cantidad de operaciones matemáticas se cuadruplica. Llegado a cierto punto (por ejemplo, al intentar procesar un libro entero), la ráfaga de cálculos requeridos vuelve al procesamiento computacionalmente inmanejable e ineficiente en tiempo.
2. Memoria (Saturación del KV Cache): Durante la etapa de generación autoregresiva (decode), el modelo almacena en la memoria física de la GPU (VRAM) los vectores de Key y Value de cada token pasado para evitar recalcular su historia una y otra vez. El tamaño de este KV cache crece de forma lineal y agresiva a medida que la secuencia de contexto se alarga. Al habilitar ventanas de contexto extremadamente grandes, el caché de un solo usuario devora rápidamente decenas de gigabytes de VRAM, causando que el sistema alcance su tope físico y colapse por falta de memoria.
3. Positional Encoding (Codificación Posicional): Dado que la arquitectura Transformer procesa toda la información de la entrada en paralelo, carece de una noción innata sobre el orden temporal de las palabras. Para que el modelo sepa dónde está cada palabra, se le inyectan Positional Embeddings (codificaciones posicionales). La forma en que estos se configuran durante el entrenamiento puede crear una barrera arquitectónica rígida:
- Posiciones Aprendidas (Learned): En modelos tradicionales como BERT o GPT-2, el modelo aprende un vector posicional estático para cada lugar específico de la secuencia. El límite de contexto aquí es un tope duro, ya que el modelo está limitado al max_position definido durante su fase de entrenamiento (por ejemplo, un máximo de 512 posiciones en BERT) y no tiene la capacidad matemática de generalizar a secuencias más largas. Si intentás procesar un token en la posición 513, el modelo fallará porque simplemente no existe un vector aprendido para esa ubicación.
- Evolución moderna (RoPE): Modelos modernos como LLaMA o Mistral utilizan codificaciones rotatorias (RoPE), que calculan nativamente la distancia relativa temporal entre los tokens mediante ángulos de rotación. Esto permite generalizar a secuencias muchísimo más largas sin estar atados a una tabla fija. Sin embargo, aunque el Positional Encoding deje de ser una barrera limitante estricta, la ventana de contexto máxima termina siendo dictada forzosamente por los topes de hardware de la memoria (el KV cache) y el cómputo (O(N2)).

Un modelo con 32 capas, 32 KV heads, d_head=128 y secuencia de 8192 tokens en fp16: ¿cuánta memoria ocupa solo el KV cache? ¿Qué pasa si servís 64 usuarios concurrentes? Para calcular el tamaño del KV cache, multiplicamos los tensores almacenados por su tamaño en bytes: Para 1 usuario:

- Se guardan 2 tensores (Key y Value) por token, capa y cabeza.
- Memoria por token: 2 × 32 (capas) × 32 (KV heads) × 128 (d_head) × 2 bytes (fp16)

= 524.288 bytes (~512 KB).

- Memoria por secuencia: 512 KB × 8.192 tokens = 4 GB de VRAM.
## ¿Qué pasa si servís 64 usuarios concurrentes?

El consumo crece linealmente con cada nuevo historial: 4 GB × 64 = 256 GB de VRAM dedicados únicamente al KV cache. A nivel práctico, el sistema colapsaría por falta de memoria. Ninguna GPU comercial individual posee 256 GB de VRAM (el estándar más alto es 80 GB). Es precisamente por este masivo cuello de botella que modelos recientes como LLaMA o Mistral adoptaron GQA (Grouped Query Attention), reduciendo drásticamente la cantidad de cabezas KV compartidas para hacer viable servir a múltiples usuarios simultáneos. Un Transformer decoder-only tiene 32 capas, dimensión del modelo d=4096, 32 attention heads (d_head=128), FFN con dimensión intermedia 11008 (sin tener en cuenta la activación), y vocabulario de 32000 tokens.

## ¿Cuántos parámetros tiene aproximadamente?

Desglosá por componente: embedding, atención (Wq, Wk, Wv, Wo), FFN, y layer norm. Para desglosar y calcular la cantidad aproximada de parámetros de este modelo, utilizaremos la arquitectura fundamental del Transformer provista en las fuentes y realizaremos las multiplicaciones matriciales correspondientes. (Aclaración: El cálculo aritmético exacto que verás a continuación es información externa basada en las estructuras y fórmulas de las diapositivas, diseñado para resolver la pregunta 128 de tu guía).

1. Capa de Embeddings (y capa de salida) La capa de embedding consiste en una matriz que mapea cada token individual de tu vocabulario a un vector denso del tamaño de la dimensión del modelo (d).
- Cálculo: 32.000 (vocabulario) × 4096 (dimensio

ˊ n d) = 131.072.000 para ˊmetros.

- (Nota: En los modelos generativos autoregresivos existe típicamente una capa de salida o "LM Head" al final que vuelve a proyectar desde la dimensión del modelo hacia el vocabulario para predecir la siguiente palabra, requiriendo otros

131.072.000 parámetros).

2. Desglose de un Bloque Transformer (Por capa individual) Cada una de las 32 capas del decodificador está compuesta principalmente por el mecanismo de Atención y la Red Feed-Forward (FFN). A. Mecanismo de Atención (Wq,Wk,Wv,Wo): Para calcular la atención, la entrada se proyecta mediante transformaciones lineales para obtener las matrices de Query (Wq), Key (Wk) y Value (Wv). Al tener 32 cabezas de dimensión 128, su dimensión total concatenada equivale exactamente a la dimensión del modelo (32×128=4096). o Wq: 4096 × 4096 = 16.777.216 parámetros. o Wk: 4096 × 4096 = 16.777.216 parámetros. o Wv: 4096 × 4096 = 16.777.216 parámetros. o Wo (Proyección lineal de salida): 4096 × 4096 = 16.777.216 parámetros. o Total Atención por capa: 4 × 16.777.216 = 67.108.864 parámetros B. Red Feed-Forward (FFN): Esta subcapa primero expande la dimensión hacia un tamaño intermedio y luego la vuelve a comprimir a la dimensión original. Siguiendo tu directiva estricta de "sin tener en cuenta la activación", calculamos el diseño de un Transformer clásico (compuesto de 2 matrices): o Matriz 1 (d→dintermedio): 4096 × 11008 = 45.088.768 parámetros. o Matriz 2 (dintermedio→d): 11008 × 4096 = 45.088.768 parámetros. o Total FFN por capa: . (Detalle vinculado a nuestra conversación: Estos hiperparámetros coinciden con el modelo LLaMA 7B. Si se incluyera la tercera matriz compuerta que exige su función de activación SwiGLU, este componente sumaría otros ~45 millones de parámetros por capa). C. Layer Normalization: Se aplica usualmente dos veces por bloque (Pre-Norm: antes de la atención y antes de la FFN). En un LayerNorm clásico, hay parámetros entrenables de escala (γ) y sesgo (β) por cada dimensión. o Cálculo: 2 (parámetros) × 4096 = 8192 por normalización. o Total LN por capa: 2 × 8192 = 16.384 parámetros
3. Cálculo Total Aproximado del Modelo
- Suma de 1 bloque individual: 67.1M (Atención) + 90.1M (FFN) + 0.016M (LN) ≈

157.3 millones de parámetros.

- Suma de las 32 capas: 32 × 157.3M ≈ 5.033 millones (5.03 Billones)
- Total final del modelo: 5.033M (Capas) + 131M (Embeddings) + 131M (LM Head)

≈ 5.295 millones de parámetros. En conclusión, bajo estas características clásicas dictadas, el modelo tiene un tamaño de aproximadamente 5.3 Billones de parámetros (o rondando los ~6.7 Billones si se contabilizara la arquitectura real de activación SwiGLU moderna). En el modelo anterior, ¿qué porcentaje de los parámetros están en las FFN vs en las capas de atención? ¿Por qué las FFN suelen tener más parámetros que la atención pese a ser conceptualmente "más simples"? Tomando los datos matemáticos calculados en el modelo de la pregunta anterior, si analizamos los componentes principales de cada bloque (ignorando la capa de normalización por su tamaño mínimo):

- Capas de Atención: ~67.1 millones de parámetros.
- Redes Feed-Forward (FFN): ~90.1 millones de parámetros.

Esto significa que, dentro del "corazón" de cada capa del Transformer, las FFN concentran aproximadamente el 57% de los parámetros, mientras que el mecanismo de atención representa el 43% restante.

## ¿Por qué las FFN suelen tener más parámetros pese a ser conceptualmente más "simples"?

(Aclaración: Las fuentes documentales indican que la FFN cuenta con un hiperparámetro crítico llamado "Intermediate Size", pero la explicación de su función técnica como "memoria" es información externa para responder el concepto de tu guía). Aunque el mecanismo de atención realiza los complejos cálculos matemáticos para cruzar la información de la secuencia, las FFN acaparan la mayor parte del tamaño del modelo por dos motivos principales:

1.
2. Expansión Dimensional (Intermediate Size): Estructuralmente, la FFN no mantiene

el tamaño original del vector (embedding). Su diseño consiste en expandir artificialmente la información hacia un tamaño de capa oculta intermedio gigantesco (en el modelo de tu ejemplo, saltando de una dimensión de 4.096 a 11.008) para luego volver a comprimirla a su tamaño original. Estas matrices de proyección tan anchas son el principal factor que infla la cantidad de parámetros. El rol de "Banco de Memoria": Existe una división del trabajo fundamental en el Transformer. La atención actúa como un enrutador inteligente que decide qué contexto mirar y mezcla las palabras. La FFN, en cambio, toma ese resultado y decide qué pensar al respecto de forma individual por token. Este enorme espacio expansivo de parámetros en la FFN funciona en la práctica como la gran memoria de conocimiento del modelo. Es allí donde se almacenan los hechos concretos, las relaciones lógicas y la comprensión del mundo asimilada durante el pre- entrenamiento, requiriendo mucha más "capacidad bruta" de almacenamiento que la atención. Si al modelo anterior le aplicás GQA con 8 KV heads en lugar de 32,

## ¿cuántos parámetros se ahorran en Wk y Wv?

¿Es un ahorro significativo respecto al total del modelo, o el beneficio principal de GQA está en otro lado? Si aplicás GQA con 8 cabezas KV en lugar de 32, la dimensión total proyectada para las Claves (K) y Valores (V) pasa de 4096 (32×128) a solo 1024 (8×128). Esto cambia el tamaño de las matrices:

- Cálculo original (32 cabezas): Las matrices y tenían un tamaño de millones de parámetros cada una. Juntas sumaban ~33.5 millones por capa.
- Cálculo con GQA (8 cabezas): Ahora tienen un tamaño de millones cada una. Juntas suman ~8.38 millones por capa.
- Ahorro total: Te ahorrás ~25.1 millones de parámetros por capa. Multiplicado por las 32 capas, el ahorro es de ~805 millones de parámetros.
## ¿Es significativo respecto al total?

Sí, es un número considerable, ya que representa un ahorro de casi el 15% del total del modelo (que pesaba aproximadamente 5.3 Billones).

## ¿El beneficio principal está ahí?

No, el ahorro de parámetros es un efecto secundario útil, pero el beneficio principal está en el uso de memoria durante la inferencia. Tal como vimos en el ejemplo donde la GPU colapsaba al intentar servir a 64 usuarios, reducir las cabezas de 32 a 8 achica exactamente a la cuarta parte el tamaño del gigantesco KV cache. Esto resuelve el mayor cuello de botella del Transformer, permitiendo una enorme escalabilidad para servir a múltiples usuarios simultáneos y habilitando ventanas de contexto muchísimo más largas sin agotar la VRAM.
