### **Contexto General: MIA305 - NLP (2026)**
El curso recorre la transición desde los modelos estadísticos de conteo hacia las arquitecturas neuronales profundas, culminando en los modelos de lenguaje de gran escala (LLMs), su alineación y el diseño de agentes.

---

### **I. Representación del Lenguaje: De Palabras a Vectores**
*   **Semántica Distribucional:** Basada en el principio de J.R. Firth (1957): "conocerás una palabra por la compañía que guarda". El significado surge del contexto de co-ocurrencia.
*   **Word Embeddings:** Representaciones numéricas densas y compactas de dimensiones fijas (hiperparámetro) que eliminan la ingeniería de atributos manual.
    *   **Estáticos (Word2Vec, GloVe):** Cada palabra tiene un único vector fijo. **Skip-gram (W2V)** entrena una red para predecir el contexto dada una palabra central. **GloVe** utiliza estadísticas globales (matriz de co-ocurrencias).
    *   **Contextuales (ELMo, BERT):** El vector de una palabra cambia dinámicamente según su oración específica, resolviendo problemas de polisemia.
*   **Tokenización:** Proceso de segmentar texto en unidades (palabras, caracteres o subwords como **BPE**). BPE balancea el tamaño del vocabulario y el largo de la secuencia.

---

### **II. Modelos de Lenguaje (LM) y Redes Recurrentes**
*   **Definición de LM:** Función que calcula la probabilidad de una secuencia de palabras o del siguiente token dado un historial.
*   **Modelos N-gram:** Estadísticos clásicos basados en una ventana fija de $N-1$ palabras previas. Sus fallas incluyen la **escasez (sparsity)** y la falta de generalización semántica.
*   **Evaluación:** La métrica intrínseca estándar es la **Perplejidad (Perplexity)**; un valor menor indica que el modelo es más seguro en sus predicciones.
*   **RNN (Redes Recurrentes):** Diseñadas para datos secuenciales manteniendo un estado oculto ($h_t$) que actúa como memoria.
    *   **Limitaciones:** Sufren del **gradiente desvaneciente (vanishing gradient)**, lo que les impide aprender dependencias a largo plazo y causa que "olviden" el inicio de secuencias largas.
    *   **LSTM (Long Short-Term Memory):** Soluciona el olvido mediante "puertas" (gates) que regulan qué información mantener o descartar.
    *   **ELMo:** Primer modelo masivo de embeddings contextuales usando LSTMs bidireccionales pre-entrenadas.

---

### **III. La Revolución del Transformer y BERT**
*   **Mecanismo de Atención:** Permite al modelo enfocarse en partes relevantes de la entrada sin importar la distancia, eliminando el "cuello de botella" de las RNN.
*   **Transformer:** Arquitectura basada puramente en atención que permite **paralelización masiva**.
    *   **Self-Attention:** Interacción de tokens mediante vectores **Query (Q), Key (K) y Value (V)**. Se escala por $1/\sqrt{d_k}$ para evitar la saturación de la función softmax.
    *   **Positional Embeddings:** Inyectan información sobre el orden de los tokens, ya que la atención no es secuencial.
*   **BERT (Encoder-only):** Pre-entrenado mediante **Masked LM (MLM)** (predecir palabras ocultas) y **NSP** (predecir la siguiente oración). Utiliza atención bidireccional profunda para tareas de comprensión como NER y clasificación. El token **[CLS]** se usa como representación agregada de toda la oración.

---

### **IV. Modelos Generativos y el Paradigma LLM**
*   **GPT (Decoder-only):** Modelo autorregresivo que predice la siguiente palabra de izquierda a derecha (unidireccional).
*   **Scaling Laws:** Establecen que el rendimiento mejora de forma predecible al escalar parámetros ($N$), datos ($D$) y cómputo ($C$).
*   **Capacidades Emergentes:** Habilidades como razonamiento o programación que aparecen abruptamente al alcanzar escalas críticas de parámetros.
*   **Alucinaciones:** Generación de texto plausible pero factualmente incorrecto debido a la naturaleza probabilística del modelo. Se mitigan con técnicas como **RAG** (Retrieval-Augmented Generation).

---

### **V. Alineación y Post-entrenamiento**
*   **Modelos Base vs. Instruct:** Los modelos base solo completan texto estadísticamente; los instruidos están alineados para seguir órdenes.
*   **Pipeline de Entrenamiento para Asistentes:**
    1.  **Pre-training (PT):** Aprendizaje no supervisado en corpus masivos.
    2.  **SFT (Supervised Fine-Tuning):** Entrenamiento con pares de instrucción-respuesta (Instruction Tuning).
    3.  **Alineación Humana (RLHF/DPO):** Optimización basada en preferencias humanas usando **RLHF** (Reinforcement Learning from Human Feedback) con un **Reward Model**, o **DPO** (Direct Preference Optimization), que es más eficiente al no requerir un modelo de recompensa separado.

---

### **VI. Eficiencia y Tópicos Avanzados**
*   **Adaptación Eficiente (LoRA):** Técnica de ajuste fino que inyecta matrices de bajo rango para adaptar modelos gigantes con pocos recursos.
*   **Inferencia y Memoria:**
    *   **KV Cache:** Almacena estados anteriores de atención para acelerar la generación, escalando linealmente con la secuencia.
    *   **GQA (Grouped Query Attention):** Reduce el tamaño del KV cache y mejora la velocidad de inferencia compartiendo llaves/valores entre cabezas.
*   **Sistemas Agénticos:** Uso de LLMs como "cerebro" para planificar, usar herramientas externas y ejecutar bucles de razonamiento (**Chain-of-Thought**, ReAct).


Para completar tu guía de estudio con el mismo nivel de detalle, aquí tenés el desarrollo del **Eje 6 (continuación) y el Eje 7 (Cálculos y Eficiencia)**, que corresponden a las preguntas finales del archivo guía (100 a 130). 

Este bloque es el más técnico y el que suele definir la nota más alta en un examen, ya que trata sobre la implementación real y los costos de los modelos.

---

### 🧵 Eje 6 (Cont.): Alineación y Razonamiento (100 - 123)
*¿Cómo pasamos de un predictor de palabras a un sistema que "piensa" y se alinea con nosotros?*

#### 13. ¿Cuál es la diferencia entre SFT (Supervised Fine-Tuning) y RLHF?
*   **Teoría:** El SFT enseña al modelo el formato (instrucción-respuesta). El RLHF lo alinea con preferencias humanas subjetivas usando un modelo de recompensa.
*   **Respuesta de Examen:** El **SFT** es un aprendizaje supervisado donde el modelo imita pares de ejemplos escritos por humanos. El **RLHF** es un proceso de tres etapas (Pre-training → Reward Modeling → Optimización con PPO/DPO) que ajusta el modelo para que sus respuestas sean preferidas por humanos en términos de utilidad, honestidad y seguridad.

#### 14. ¿Qué es el Chain-of-Thought (CoT) y por qué mejora el razonamiento?
*   **Teoría:** Es una técnica de prompting (o entrenamiento) que fuerza al modelo a descomponer problemas complejos en pasos lógicos.
*   **Respuesta de Examen:** El **Chain-of-Thought** consiste en inducir al modelo a generar una secuencia de pasos intermedios de razonamiento antes de dar la respuesta final. Esto mejora el rendimiento en tareas aritméticas y lógicas porque permite al modelo utilizar más tokens (computación de paso de tiempo) para procesar la información antes de llegar a la conclusión.

---

### 🧵 Eje 7: Arquitectura Avanzada, Memoria y Eficiencia (124 - 130)
*La "fontanería" de los LLMs: ¿Por qué son tan caros de ejecutar?*

#### 15. ¿Qué sucede si duplicamos el largo de contexto (ej. de 4K a 8K tokens)?
*   **Teoría:** El costo de la atención es cuadrático, pero el almacenamiento de los estados previos (KV Cache) es lineal.
*   **Respuesta de Examen:** Al duplicar el contexto, el uso de memoria del **KV Cache** aumenta de forma **lineal** ($O(N)$). Sin embargo, el costo computacional de la **atención** aumenta de forma **cuadrática** ($O(N^2)$), ya que cada nuevo token debe atender a todos los anteriores, cuadriplicando las operaciones de producto escalar.

#### 16. ¿Qué es el GQA (Grouped Query Attention) y cuál es su beneficio?
*   **Teoría:** Es un punto medio entre la atención estándar (MHA) y la atención de una sola cabeza para las llaves (MQA).
*   **Respuesta de Examen:** **GQA** divide las cabezas de las queries en grupos, y cada grupo comparte una única cabeza de **Key** y **Value**. Su beneficio principal es reducir drásticamente el tamaño del **KV Cache** en memoria VRAM, lo que permite procesar secuencias mucho más largas y aumentar el *throughput* (usuarios concurrentes) sin una pérdida significativa de calidad en comparación con Multi-Head Attention.

#### 17. ¿Por qué las FFN (Feed-Forward Networks) suelen tener más parámetros que la atención?
*   **Teoría:** La atención es para "comunicar" tokens; la FFN es para "procesar" el conocimiento de cada token por separado.
*   **Respuesta de Examen:** En un Transformer *decoder-only*, aproximadamente **2/3 de los parámetros** residen en las capas **FFN**. Esto se debe a que la FFN expande la dimensionalidad del modelo (típicamente 4 veces la dimensión $d$) para actuar como una memoria de almacenamiento de conocimiento estático, mientras que la atención solo requiere matrices de proyección para las cabezas, siendo más eficiente en parámetros pero más costosa en memoria durante la inferencia por el cache.

---

### 💡 Última recomendación de tu profesor
Para el cálculo de memoria (pregunta 127): Recordá que cada parámetro en **fp16** ocupa **2 bytes**. Si tenés que calcular el KV Cache, la fórmula es: $2 \times \text{capas} \times \text{cabezas} \times d_{\text{head}} \times \text{largo de secuencia} \times 2 \text{ bytes}$.


