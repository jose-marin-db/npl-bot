import telebot
import random
import time
import json
import os
import threading
import hashlib
import dotenv
from groq import Groq, RateLimitError

dotenv.load_dotenv()

TOKEN = os.getenv("token")

bot = telebot.TeleBot(TOKEN)
groq_client = Groq(api_key=os.getenv("GROQ_TOKEN"))

# Modelo principal y respaldo si se agota la cuota (TPD) del tier gratuito / on_demand.
# Ver límites en https://console.groq.com/docs/rate-limits
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")


def groq_chat_completion(
    *,
    messages: list,
    temperature: float,
    max_tokens: int,
    model: str | None = None,
):
    """
    Llama a chat.completions; ante rate limit / cuota diaria del modelo principal,
    reintenta una vez con GROQ_MODEL_FALLBACK (suele tener cupo aparte).
    """
    primary = model or GROQ_MODEL
    try:
        return groq_client.chat.completions.create(
            model=primary,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except RateLimitError as e:
        fb = (GROQ_MODEL_FALLBACK or "").strip()
        if fb and primary != fb:
            print(f"[Groq] Límite en {primary!r} ({e}); reintentando con {fb!r}")
            return groq_client.chat.completions.create(
                model=fb,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise

# Cargamos el contexto de clase una sola vez al arrancar
_RUTA_CONTEXTO = os.path.join(os.path.dirname(__file__), "contexto-clase.md")
with open(_RUTA_CONTEXTO, encoding="utf-8") as _f:
    CONTEXTO_CLASE = _f.read()



# Temas que justifican una explicación matemática profunda
_TEMAS_MATEMATICOS = [
    "atención", "attention", "query", "key", "value", "softmax", "transformer",
    "embedding", "vector", "matriz", "matrix", "dimensi", "producto escalar",
    "dot product", "gradiente", "backprop", "lstm", "gru", "rnn", "perplexity",
    "perplejidad", "log", "probabilidad", "distribución", "normal", "relu",
    "activación", "layer norm", "positional", "kv cache", "lora", "rango",
    "parámetro", "gqa", "ffn", "feed-forward", "cabeza", "head", "escala",
    "n-gram", "bigram", "unigram", "word2vec", "glove", "skip-gram", "elmo",
    "bert", "gpt", "fine-tuning", "sft", "rlhf", "dpo", "reward", "bpe",
]

def _requiere_matematica(q: dict) -> bool:
    """Determina si el tema de la pregunta justifica una explicación con fórmulas."""
    texto = (q["pregunta"] + " " + q["explicacion"]).lower()
    return any(keyword in texto for keyword in _TEMAS_MATEMATICOS)

def generar_teoria_profunda(q: dict) -> str:
    """
    Genera una explicación técnica detallada con fórmulas y notación matricial
    cuando el tema lo requiere. Para temas conceptuales, profundiza en la intuición.
    """
    con_matematica = _requiere_matematica(q)
    respuesta_correcta = q["opciones"][q["correcta_id"]]

    if con_matematica:
        instruccion_formato = """Incluí:
- La fórmula o expresión matemática central del concepto.
- Qué representa cada variable o dimensión.
- Una intuición de por qué la ecuación funciona así.
- Si hay operaciones matriciales, describí las dimensiones de entrada y salida.

REGLAS DE FORMATO OBLIGATORIAS (el texto se muestra en Telegram, NO soporta LaTeX):
- PROHIBIDO usar LaTeX: no uses \\frac, \\sqrt, \\mathbf, \\sum, $...$ ni ninguna sintaxis LaTeX.
- Para fracciones escribí: a/b  (ejemplo: 1/√dₖ)
- Para raíces escribí: √x
- Para subíndices usá letras directas o sufijos: d_k, h_t, W_Q
- Para potencias escribí: x² o x^2
- Para sumatorias: Σᵢ o suma(i=1 a n)
- Para vectores/matrices: notación simple como v ∈ ℝ^d o W ∈ ℝ^(dₖ × d)
- Usá símbolos Unicode si ayudan: ∈ ℝ ∑ √ · × → ≈ ≤ ≥"""
    else:
        instruccion_formato = """Incluí:
- Por qué este concepto es relevante en el pipeline de NLP/LLMs.
- Cómo se relaciona con otros conceptos del curso.
- Un ejemplo concreto o analogía que lo haga más intuitivo.

REGLAS DE FORMATO: texto plano, sin LaTeX, sin símbolos raros. Solo caracteres que se lean bien en chat."""

    prompt = f"""Sos un profesor experto del curso MIA305 - NLP, con profundo conocimiento técnico. Un alumno quiere entender en profundidad el siguiente concepto.

Pregunta: {q['pregunta']}
Respuesta correcta: {respuesta_correcta}
Explicación base: {q['explicacion']}

Material de clase de referencia:
---
{CONTEXTO_CLASE}
---

{instruccion_formato}

Escribí una explicación técnica y profunda (máximo 6 oraciones o bloques). Usá un tono de profesor que domina el tema. Sin saludos ni introducciones. Solo la explicación."""

    try:
        respuesta = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] Error al generar teoría profunda: {e}")
        return q["explicacion"]

def generar_explicacion(q: dict, acerto: bool) -> str:
    """
    Usa Groq + el contexto de la clase para generar una explicación personalizada
    y más rica que la estática del dataset.
    Si falla, devuelve la explicación original.
    """
    estado = "respondió CORRECTAMENTE" if acerto else "respondió INCORRECTAMENTE"
    respuesta_correcta = q["opciones"][q["correcta_id"]]

    prompt = f"""Sos un tutor experto del curso MIA305 - NLP. Un alumno {estado} la siguiente pregunta de opción múltiple.

Pregunta: {q['pregunta']}
Respuesta correcta: {respuesta_correcta}
Explicación base: {q['explicacion']}

Usá el siguiente material de clase como referencia para enriquecer la explicación:
---
{CONTEXTO_CLASE}
---

Escribí una explicación breve (máximo 4 oraciones) que:
- Confirme por qué la respuesta correcta es correcta.
- Conecte el concepto con el material del curso si es relevante.
- Si el alumno se equivocó, aclare por qué las otras opciones son incorrectas de forma concisa.
- Use un tono didáctico y directo, sin saludos ni introducciones.

FORMATO: texto plano legible en Telegram. PROHIBIDO usar LaTeX (\\frac, \\sqrt, $...$, etc.). Si necesitás expresar matemática usá: √, ², ∈, ℝ, Σ, ·, → y notación simple como a/b o x^2."""

    try:
        respuesta = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] Error al generar explicación: {e}")
        return q["explicacion"]  # Fallback a la explicación original

def mejorar_pregunta(q: dict) -> dict:
    """
    Llama a Groq para reescribir el enunciado y las opciones con un estilo
    más natural y variado, conservando la respuesta correcta y la explicación.
    Si la llamada falla, devuelve la pregunta original sin modificar.
    """
    correcta_texto = q["opciones"][q["correcta_id"]]

    prompt = f"""Sos un experto en NLP y Machine Learning. Vas a mejorar una pregunta de opción múltiple sobre MIA (Maestría en IA Aplicada) para que suene más natural, clara y desafiante. 

Reglas estrictas:
- Mantené exactamente 4 opciones.
- No cambies el significado correcto de la respuesta.
- No cambies la explicación.
- Todas las opciones deben tener una longitud similar.
- Devolvé SOLO un JSON válido con este formato exacto, sin texto adicional:
{{
  "pregunta": "...",
  "opciones": ["...", "...", "...", "..."],
  "correcta_texto": "..."
}}

Pregunta original: {q['pregunta']}
Opciones originales: {json.dumps(q['opciones'], ensure_ascii=False)}
Respuesta correcta: {correcta_texto}
"""

    try:
        respuesta = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        contenido = respuesta.choices[0].message.content.strip()

        # Extraemos solo el bloque JSON por si Groq agrega texto extra
        inicio = contenido.find("{")
        fin = contenido.rfind("}") + 1
        datos = json.loads(contenido[inicio:fin])

        # Recalculamos correcta_id según el texto que Groq marcó como correcto
        # Primero buscamos igualdad exacta; si no, buscamos por substring como fallback
        correcta_buscada = datos["correcta_texto"].strip()
        nuevo_correcta_id = next(
            (i for i, op in enumerate(datos["opciones"]) if op.strip() == correcta_buscada),
            None
        )
        if nuevo_correcta_id is None:
            nuevo_correcta_id = next(
                (i for i, op in enumerate(datos["opciones"]) if correcta_buscada in op),
                None
            )
        if nuevo_correcta_id is None:
            print("[Groq] No se pudo identificar la opción correcta. Usando pregunta original.")
            return q

        return {
            "pregunta": datos["pregunta"],
            "opciones": datos["opciones"],
            "correcta_id": nuevo_correcta_id,
            "explicacion": q["explicacion"],  # La explicación nunca cambia
        }

    except Exception as e:
        print(f"[Groq] Error al mejorar pregunta: {e}")
        return q  # Fallback a la pregunta original

# Tu lista de 140 preguntas
preguntas = [
    {
        "pregunta": "¿Qué es transfer Learning?",
        "opciones": ["Entrenar un modelo desde cero para cada tarea", "Reutilizar representaciones aprendidas en tareas generales para tareas específicas", "Un método para reducir el tamaño del vocabulario", "Una técnica de limpieza de datos manual"],
        "correcta_id": 1,
        "explicacion": "El Transfer Learning consiste en aprovechar el conocimiento (pesos y representaciones) de un modelo entrenado en un corpus masivo para resolver tareas particulares con menos datos."
    },
    {
        "pregunta": "¿Qué es la semántica distribucional?",
        "opciones": ["Definir palabras mediante sus raíces latinas", "El significado de una palabra según su contexto de co-ocurrencia", "Un sistema de reglas gramaticales fijas", "La traducción literal entre dos idiomas"],
        "correcta_id": 1,
        "explicacion": "Se basa en el principio de J.R. Firth: 'conocerás una palabra por la compañía que guarda', es decir, por las palabras que aparecen cerca de ella."
    },
    {
        "pregunta": "¿Por qué son importantes los embeddings?",
        "opciones": ["Porque son vectores fáciles de leer para humanos", "Porque eliminan la necesidad de ingeniería de atributos manual", "Porque ocupan más memoria que el one-hot encoding", "Porque solo funcionan con diccionarios predefinidos"],
        "correcta_id": 1,
        "explicacion": "Los embeddings aprenden automáticamente las características relevantes de los datos, transformando el lenguaje en vectores numéricos densos que las redes neuronales pueden procesar eficientemente."
    },
    {
        "pregunta": "¿Qué es un embedding?",
        "opciones": ["Un vector ralo de alta dimensionalidad", "Una representación numérica densa de dimensiones fijas", "Una lista de sinónimos de WordNet", "Un error en el proceso de entrenamiento"],
        "correcta_id": 1,
        "explicacion": "Es una representación vectorial compacta donde la información está distribuida en todos sus componentes, aprendida de forma no supervisada."
    },
    {
        "pregunta": "¿Qué son embeddings contextuales?",
        "opciones": ["Vectores que no cambian según la oración", "Vectores que dependen de la posición y entorno de la palabra", "Representaciones basadas únicamente en la frecuencia de la palabra", "Embeddings que solo se usan en diccionarios"],
        "correcta_id": 1,
        "explicacion": "A diferencia de los estáticos, estos generan un vector distinto para palabras polisémicas (ej. 'banco') dependiendo de su uso en la oración."
    },
    {
        "pregunta": "¿Cómo se implementa el transfer learning en una red neuronal?",
        "opciones": ["Cambiando la función de activación aleatoriamente", "Mediante las etapas de pre-entrenamiento y ajuste fino (fine-tuning)", "Eliminando las capas ocultas del modelo", "Usando solo datos etiquetados desde el inicio"],
        "correcta_id": 1,
        "explicacion": "Primero se pre-entrena en un corpus masivo sin etiquetas y luego se ajustan los pesos para una tarea específica con datos etiquetados."
    },
    {
        "pregunta": "¿Cuáles son las ventajas del transfer learning en términos de datos?",
        "opciones": ["Requiere más datos etiquetados que el entrenamiento normal", "Permite obtener alto rendimiento con datasets pequeños etiquetados", "No requiere ningún dato para el ajuste fino", "Solo funciona si los datos son imágenes"],
        "correcta_id": 1,
        "explicacion": "Al partir de pesos cercanos al óptimo, reduce drásticamente la dependencia de grandes cantidades de datos manualmente etiquetados."
    },
    {
        "pregunta": "¿Qué interpretación tiene que dos palabras estén 'cerca' en un espacio vectorial?",
        "opciones": ["Que tienen la misma cantidad de letras", "Que comparten significados o contextos similares", "Que aparecen siempre al principio de las oraciones", "Que pertenecen al mismo idioma"],
        "correcta_id": 1,
        "explicacion": "La cercanía matemática (distancia corta) en el espacio de embeddings refleja similitud semántica y funcional entre las palabras."
    },
    {
        "pregunta": "¿Qué es el método Skip-Gram en Word2Vec?",
        "opciones": ["Predecir el contexto dada una palabra central", "Predecir la palabra central dado el contexto", "Contar la frecuencia global de las palabras", "Agrupar palabras por su longitud"],
        "correcta_id": 0,
        "explicacion": "Skip-Gram entrena al modelo para maximizar la probabilidad de observar las palabras de la ventana de contexto basándose en una palabra de entrada."
    },
    {
        "pregunta": "¿Qué es el método GloVe?",
        "opciones": ["Un modelo basado solo en ventanas locales", "Un modelo de vectores globales basado en matrices de co-ocurrencia", "Una técnica de traducción automática", "Un algoritmo de clustering para documentos"],
        "correcta_id": 1,
        "explicacion": "GloVe aprovecha las estadísticas globales de todo el corpus para optimizar las representaciones vectoriales."
    },
    {
        "pregunta": "¿Cuál es la diferencia conceptual entre Word2Vec y GloVe?",
        "opciones": ["W2V es global y GloVe es local", "W2V es un modelo de contexto local y GloVe usa estadísticas globales", "GloVe solo funciona para inglés y W2V para todos los idiomas", "No hay diferencia conceptual"],
        "correcta_id": 1,
        "explicacion": "Word2Vec itera sobre ventanas de texto individuales (local), mientras que GloVe utiliza la matriz de co-ocurrencias de todo el corpus (global)."
    },
    {
        "pregunta": "¿Qué es la tokenización?",
        "opciones": ["Eliminar las palabras frecuentes de un texto", "Dividir el texto en unidades mínimas llamadas tokens", "Convertir palabras en vectores numéricos directamente", "Corregir la ortografía de un documento"],
        "correcta_id": 1,
        "explicacion": "Es el proceso de segmentación de un flujo de texto en piezas manejables, como palabras, fragmentos o caracteres."
    },
    {
        "pregunta": "¿Qué es un modelo de lenguaje?",
        "opciones": ["Un software de traducción", "Una función que calcula la probabilidad de una secuencia de palabras", "Un diccionario electrónico", "Un corrector de gramática"],
        "correcta_id": 1,
        "explicacion": "Su objetivo es asignar probabilidades a secuencias de tokens o predecir el siguiente token dado un historial."
    },
    {
        "pregunta": "¿Qué es un modelo de lenguaje unidireccional?",
        "opciones": ["Aquel que solo lee de derecha a izquierda", "Aquel que predice el token actual basándose solo en el pasado", "Aquel que procesa toda la oración al mismo tiempo", "Un modelo que no usa atención"],
        "correcta_id": 1,
        "explicacion": "Es un modelo causal que respeta el orden temporal, sin 'ver el futuro' durante el entrenamiento o generación."
    },
    {
        "pregunta": "¿Qué es un modelo de lenguaje generativo?",
        "opciones": ["Un modelo que solo clasifica textos", "Un modelo capaz de producir nuevas secuencias de texto", "Un algoritmo que genera imágenes a partir de tablas", "Un sistema que elimina palabras raras"],
        "correcta_id": 1,
        "explicacion": "Son modelos entrenados para modelar la distribución de los datos y crear contenido nuevo de forma autorregresiva."
    },
    {
        "pregunta": "¿Qué es el proceso autorregresivo en modelos generativos?",
        "opciones": ["Generar todos los tokens en paralelo", "Usar el token generado anteriormente como entrada para predecir el siguiente", "Corregir automáticamente los errores de la red", "Entrenar el modelo sin intervención humana"],
        "correcta_id": 1,
        "explicacion": "La generación se realiza paso a paso, retroalimentando cada salida al contexto para mantener la coherencia."
    },
    {
        "pregunta": "¿Qué es un modelo de lenguaje n-gram?",
        "opciones": ["Un modelo neuronal profundo", "Un modelo estadístico que mira una ventana fija de N-1 palabras previas", "Un modelo que no tiene vocabulario fijo", "Un tipo de embedding contextual"],
        "correcta_id": 1,
        "explicacion": "Es una aproximación de Markov que simplifica la predicción limitando el contexto a un número fijo de palabras anteriores."
    },
    {
        "pregunta": "¿Cómo se entrena un modelo n-gram?",
        "opciones": ["Mediante backpropagation", "Contando frecuencias de secuencias en un corpus", "Usando el mecanismo de atención", "Mediante refuerzo humano"],
        "correcta_id": 1,
        "explicacion": "Se basa en el conteo de apariciones de secuencias de tamaño N y el cálculo de probabilidades relativas."
    },
    {
        "pregunta": "¿Cuáles son las principales desventajas de los n-gramas?",
        "opciones": ["Son demasiado lentos", "Sparsity (escasez de datos) y falta de generalización semántica", "Requieren GPUs muy potentes", "Solo funcionan para oraciones cortas"],
        "correcta_id": 1,
        "explicacion": "Sufren cuando una secuencia no aparece en el entrenamiento (probabilidad 0) y no entienden la similitud entre palabras (ej. gato/felino)."
    },
    {
        "pregunta": "¿Cuál es la importancia del vocabulario?",
        "opciones": ["Determina la velocidad de la red", "Define el espacio de búsqueda y la cobertura del lenguaje del modelo", "No es importante si se usa Deep Learning", "Sirve solo para la lematización"],
        "correcta_id": 1,
        "explicacion": "El vocabulario limita qué palabras puede 'ver' y generar el modelo; un mal diseño genera demasiados tokens desconocidos (<UNK>)."
    },
    {
        "pregunta": "¿Qué es la lematización?",
        "opciones": ["Eliminar los signos de puntuación", "Reducir una palabra a su forma base o de diccionario (lema)", "Convertir todo el texto a minúsculas", "El proceso de asignar vectores a los tokens"],
        "correcta_id": 1,
        "explicacion": "Es una técnica de normalización que agrupa distintas formas flexionadas de una palabra para que el modelo las trate como una sola unidad."
    },
    {
        "pregunta": "¿Qué ventajas trae la lematización?",
        "opciones": ["Aumenta el tamaño del vocabulario", "Reduce la dimensionalidad y ayuda a mitigar la escasez de datos", "Hace que el modelo sea más creativo", "Permite capturar mejor el sarcasmo"],
        "correcta_id": 1,
        "explicacion": "Al reducir la variabilidad léxica, el modelo puede aprender mejor las estadísticas de palabras que antes estaban dispersas."
    },
    {
        "pregunta": "¿Cuándo conviene lematizar?",
        "opciones": ["Siempre en todos los casos", "En modelos de lenguaje clásicos (n-grams) con pocos datos", "En modelos generativos modernos como GPT", "Nunca, ya no se utiliza"],
        "correcta_id": 1,
        "explicacion": "Es útil en modelos estadísticos clásicos para reducir el 'sparsity', pero en modelos modernos se prefiere preservar la morfología completa."
    },
    {
        "pregunta": "¿Qué es F1-Score?",
        "opciones": ["La velocidad de inferencia del modelo", "La media armónica entre Precision y Recall", "El porcentaje de aciertos totales", "La cantidad de parámetros del Transformer"],
        "correcta_id": 1,
        "explicacion": "Es una métrica balanceada que es especialmente útil cuando las clases en el dataset están desbalanceadas."
    },
    {
        "pregunta": "¿Qué tarea requiere maximizar el Recall?",
        "opciones": ["Detección de spam (evitar falsos positivos)", "Detección de enfermedades críticas (evitar falsos negativos)", "Traducción automática", "Clasificación de sentimientos en redes sociales"],
        "correcta_id": 1,
        "explicacion": "En medicina es vital no pasar por alto ningún caso positivo, aunque eso implique algunos falsos positivos."
    },
    {
        "pregunta": "¿Cómo se evalúan intrínsecamente los modelos de lenguaje?",
        "opciones": ["Mediante el Accuracy en clasificación", "Utilizando la métrica de Perplejidad (Perplexity)", "Preguntando a expertos humanos", "Midiendo el tiempo de respuesta"],
        "correcta_id": 1,
        "explicacion": "La evaluación intrínseca mide qué tan bien el modelo predice una muestra de testeo sin aplicarlo a una tarea final."
    },
    {
        "pregunta": "¿Qué es perplexity?",
        "opciones": ["El tiempo que tarda el modelo en confundirse", "La inversa de la probabilidad de una secuencia, normalizada por su longitud", "Un tipo de capa en los Transformers", "La cantidad de palabras desconocidas en un texto"],
        "correcta_id": 1,
        "explicacion": "Mide la incertidumbre del modelo; una perplejidad baja indica que el modelo es más 'seguro' en sus predicciones."
    },
    {
        "pregunta": "¿Por qué los embeddings y las redes neuronales 'maridan' bien?",
        "opciones": ["Porque ambos son de código abierto", "Porque las redes requieren inputs numéricos continuos que los embeddings proveen", "Porque se inventaron el mismo año", "Porque no requieren entrenamiento"],
        "correcta_id": 1,
        "explicacion": "Los embeddings sitúan a las palabras en un espacio continuo donde la red neuronal puede aprender relaciones de similitud matemática."
    },
    {
        "pregunta": "En una red feed forward para clasificación de texto, ¿cómo se representa la oración?",
        "opciones": ["Usando solo la primera palabra", "Concatenando o promediando los embeddings de sus palabras", "Usando una matriz de adyacencia", "Mediante un estado oculto recurrente"],
        "correcta_id": 1,
        "explicacion": "Para alimentar una capa densa fija, se suele comprimir la información de los tokens en un único vector de representación de oración [Clase 2]."
    },
    {
        "pregunta": "¿Qué es BPE (Byte Pair Encoding)?",
        "opciones": ["Un algoritmo de cifrado de seguridad", "Una técnica de subword tokenization basada en fusiones frecuentes", "Un modelo de lenguaje de n-gramas", "Una forma de reducir la tasa de aprendizaje"],
        "correcta_id": 1,
        "explicacion": "Permite manejar palabras fuera de vocabulario dividiéndolas en fragmentos conocidos, balanceando el tamaño del vocabulario."
    },
    {
        "pregunta": "¿Qué son las redes neuronales recurrentes (RNN)?",
        "opciones": ["Redes que no tienen memoria del pasado", "Arquitecturas que procesan secuencias manteniendo un estado oculto", "Modelos que solo procesan imágenes", "Redes que se entrenan en un solo paso"],
        "correcta_id": 1,
        "explicacion": "Están diseñadas para datos secuenciales donde la salida actual depende del input presente y de la historia pasada."
    },
    {
        "pregunta": "¿Qué elementos componen una célula recurrente básica?",
        "opciones": ["Solo una función Softmax", "Un estado oculto y pesos de transición temporal", "Tres puertas de control (entrada, olvido, salida)", "Un mecanismo de atención"],
        "correcta_id": 1,
        "explicacion": "La célula básica utiliza el input actual y el estado oculto anterior para calcular el nuevo estado oculto."
    },
    {
        "pregunta": "¿Qué ventaja tienen las RNN frente a los n-gramas?",
        "opciones": ["Son más simples de entrenar", "Pueden, en teoría, manejar dependencias de largo plazo y secuencias de longitud variable", "No requieren vectores numéricos", "Son más rápidas para paralelizar"],
        "correcta_id": 1,
        "explicacion": "A diferencia de la ventana fija de los n-gramas, las RNN mantienen una memoria que fluye a través de toda la secuencia."
    },
    {
        "pregunta": "¿A qué se refiere el término bidireccional en las redes recurrentes?",
        "opciones": ["Que la red puede procesar imágenes y texto", "Que procesa la secuencia desde el principio al final y desde el final al principio simultáneamente", "Que el modelo puede traducir entre dos idiomas a la vez", "Que la red tiene el doble de neuronas en cada capa"],
        "correcta_id": 1,
        "explicacion": "Las RNR bidireccionales constan de dos redes cuyos outputs se concatenan para capturar el contexto tanto pasado como futuro de cada token."
    },
    {
        "pregunta": "¿Por qué una red bidireccional no sirve para un modelo de lenguaje puro?",
        "opciones": ["Porque es demasiado lenta para entrenar", "Porque tiene 'información de más' (del futuro) para la tarea de predicción causal", "Porque no utiliza vectores de embeddings", "Porque solo funciona con secuencias muy cortas"],
        "correcta_id": 1,
        "explicacion": "En un modelo de lenguaje se busca predecir el siguiente token; una red bidireccional ya 'conoce' las palabras de la derecha, invalidando la naturaleza de la predicción."
    },
    {
        "pregunta": "¿En qué tareas es especialmente efectiva la bidireccionalidad?",
        "opciones": ["Generación de cuentos", "Traducción, etiquetado (NER) o clasificación donde ya se tiene la secuencia completa", "Cálculo de probabilidades n-gram", "Compresión de archivos de texto"],
        "correcta_id": 1,
        "explicacion": "Es ideal para tareas donde el contexto global es necesario y la secuencia de entrada ya está disponible en su totalidad."
    },
    {
        "pregunta": "¿Qué son los embeddings contextuales?",
        "opciones": ["Vectores que se buscan en un diccionario fijo", "Representaciones dinámicas que cambian según el contexto específico de la palabra en una oración", "Vectores que solo contienen información sintáctica", "Representaciones binarias de las palabras"],
        "correcta_id": 1,
        "explicacion": "A diferencia de los estáticos, permiten que una palabra como 'banco' tenga vectores distintos según si se refiere a dinero o a un asiento."
    },
    {
        "pregunta": "¿Qué es ELMo?",
        "opciones": ["Un algoritmo de búsqueda en Google", "Un modelo de lenguaje bidireccional profundo que genera embeddings contextuales", "Una técnica de limpieza de stopwords", "Un tipo de red neuronal feed-forward simple"],
        "correcta_id": 1,
        "explicacion": "ELMo genera representaciones dinámicas procesando el texto con LSTMs bidireccionales de varias capas."
    },
    {
        "pregunta": "¿Cuál es la arquitectura principal de ELMo?",
        "opciones": ["Un Transformer de 12 capas", "LSTMs bidireccionales con embeddings basados en caracteres (CNN)", "Una red recurrente simple con una sola capa oculta", "Una matriz de co-ocurrencia global"],
        "correcta_id": 1,
        "explicacion": "Utiliza una arquitectura de LSTMs bidireccionales de varias capas que recibe como entrada embeddings de caracteres para ser más robusto."
    },
    {
        "pregunta": "¿Qué rol juegan las capas inferiores vs. superiores en ELMo?",
        "opciones": ["Las inferiores son para imágenes y las superiores para texto", "Las inferiores capturan sintaxis (gramática) y las superiores capturan semántica (significado)", "No hay diferencia entre las capas", "Las superiores se encargan de la tokenización"],
        "correcta_id": 1,
        "explicacion": "Las capas bajas de las LSTMs se centran en la estructura gramatical, mientras que las altas refinan el significado y las relaciones semánticas."
    },
    {
        "pregunta": "¿Cómo se entrena ELMo?",
        "opciones": ["Mediante clasificación de imágenes", "Prediciendo la siguiente palabra en un corpus masivo (como Wikipedia) de forma bidireccional", "Con etiquetas manuales de sentimientos", "Contando frecuencias de bigramas"],
        "correcta_id": 1,
        "explicacion": "Se pre-entrena en grandes volúmenes de texto para aprender un conocimiento general del lenguaje antes del ajuste fino."
    },
    {
        "pregunta": "¿Qué beneficio principal trae el uso de modelos pre-entrenados como ELMo?",
        "opciones": ["Hacen que el modelo sea más pesado", "Permiten alcanzar mejores resultados requiriendo menos datos etiquetados para la tarea final", "Eliminan la necesidad de usar GPUs", "Solo sirven para el idioma inglés"],
        "correcta_id": 1,
        "explicacion": "Al transferir el conocimiento del pre-entrenamiento, se logra una convergencia más rápida y mejor generalización con pocos datos."
    },
    {
        "pregunta": "¿Qué es el mecanismo de atención?",
        "opciones": ["Un sistema que apaga las neuronas innecesarias", "Una técnica que permite al modelo enfocarse en partes específicas de la entrada para generar una salida", "Un método para reducir el tamaño de los embeddings", "Una regla que obliga al modelo a leer más despacio"],
        "correcta_id": 1,
        "explicacion": "La atención permite que el modelo asigne pesos a distintos tokens de la entrada según su relevancia para el estado actual del procesamiento."
    },
    {
        "pregunta": "¿Por qué surgió la necesidad del mecanismo de atención en redes recurrentes?",
        "opciones": ["Porque las RNR eran demasiado rápidas", "Para solucionar el 'cuello de botella' informativo de usar un único vector de tamaño fijo para representar secuencias largas", "Porque no existían los embeddings estáticos", "Para poder usar procesadores más viejos"],
        "correcta_id": 1,
        "explicacion": "En arquitecturas seq2seq, forzar a que toda la historia se comprima en un solo vector final (hidden state) generaba pérdida de información."
    },
    {
        "pregunta": "¿Qué es el 'cuello de botella' en el contexto de seq2seq?",
        "opciones": ["Un error en la conexión a internet", "La limitación de representar una oración entera en un único vector de tamaño fijo antes de decodificar", "La falta de memoria RAM en el servidor", "Un tipo de función de activación"],
        "correcta_id": 1,
        "explicacion": "Es el punto crítico donde el codificador debe resumir todo el contenido en un solo vector h_t, lo cual es ineficiente para oraciones largas."
    },
    {
        "pregunta": "¿Qué es el Transformer?",
        "opciones": ["Una mejora de la red LSTM", "Una arquitectura basada puramente en mecanismos de atención, eliminando la recurrencia", "Un modelo de n-gramas avanzado", "Un robot que clasifica texto"],
        "correcta_id": 1,
        "explicacion": "El Transformer rompe con la secuencialidad de las RNN, permitiendo un procesamiento paralelo masivo mediante auto-atención."
    },
    {
        "pregunta": "¿Cuáles son los componentes de un bloque del Encoder en un Transformer?",
        "opciones": ["Recurrencia y convolución", "Atención Multi-Head y una Red Feed-Forward (FFN)", "LSTMs y capas Softmax", "Solo una capa de embeddings"],
        "correcta_id": 1,
        "explicacion": "Cada bloque del encoder refina la representación del embedding a través de operaciones de atención y redes densas."
    },
    {
        "pregunta": "¿Qué es la auto-atención (self-attention)?",
        "opciones": ["Que el modelo se evalúa a sí mismo", "Un mecanismo donde cada token de una secuencia interactúa con todos los demás de la misma secuencia", "Un filtro que elimina las palabras raras", "Una función que aumenta el valor de los pesos"],
        "correcta_id": 1,
        "explicacion": "Permite que el modelo entienda la relación de cada palabra con su contexto dentro de la misma oración."
    },
    {
        "pregunta": "¿Cuál es la intuición detrás de Query (Q), Key (K) y Value (V)?",
        "opciones": ["Son variables aleatorias sin significado", "Q es lo que busco, K es la etiqueta de la información y V es la información en sí", "Representan el pasado, presente y futuro", "Son los pesos de una red recurrente"],
        "correcta_id": 1,
        "explicacion": "El modelo calcula la similitud entre Q y K para decidir cuánta importancia (peso) darle a cada valor V."
    },
    {
        "pregunta": "¿Por qué es necesaria la escala 1/√dk en el cálculo de la atención?",
        "opciones": ["Para que el modelo sea más pequeño", "Para evitar que el producto escalar crezca demasiado y sature la función Softmax", "Para acelerar el cálculo de los logaritmos", "Para corregir errores de tipeo"],
        "correcta_id": 1,
        "explicacion": "Si los valores del producto escalar son muy grandes, el gradiente de la función softmax se vuelve casi cero, dificultando el aprendizaje."
    },
    {
        "pregunta": "¿Qué ventaja tiene el procesamiento en paralelo del Transformer frente a las RNN?",
        "opciones": ["Usa menos datos", "Permite aprovechar mejor el hardware (GPUs) al no depender de pasos secuenciales", "Es más fácil de explicar a humanos", "No requiere pre-entrenamiento"],
        "correcta_id": 1,
        "explicacion": "Al no procesar las palabras una por una, puede entrenarse en corpus masivos en mucho menos tiempo."
    },
    {
        "pregunta": "¿Qué son los positional embeddings?",
        "opciones": ["Vectores que guardan el significado de las palabras", "Vectores que inyectan información sobre el orden de los tokens en la secuencia", "Un tipo de memoria a largo plazo", "La salida final del Decoder"],
        "correcta_id": 1,
        "explicacion": "Como el Transformer procesa todo en paralelo, necesita estos embeddings para saber qué palabra va antes o después."
    },
    {
        "pregunta": "¿Para qué sirve el Layer Normalization en el Transformer?",
        "opciones": ["Para reducir la cantidad de palabras", "Para estabilizar el entrenamiento y acelerar la convergencia", "Para traducir a minúsculas", "Para eliminar los sesgos humanos"],
        "correcta_id": 1,
        "explicacion": "Normaliza las entradas de cada capa para que operen en rangos similares, mejorando la estabilidad de los gradientes."
    },
    {
        "pregunta": "¿Qué es el enmascaramiento (masking) en el Decoder del Transformer?",
        "opciones": ["Ocultar la identidad de los usuarios", "Impedir que el modelo 'vea' los tokens futuros durante el entrenamiento", "Eliminar las palabras poco frecuentes", "Codificar el texto en un lenguaje secreto"],
        "correcta_id": 1,
        "explicacion": "Asegura que la predicción de una palabra en la posición t solo dependa de los tokens en posiciones anteriores (t-1)."
    },
    {
        "pregunta": "¿Qué es BERT?",
        "opciones": ["Un modelo generativo puro", "Un modelo basado en el Encoder del Transformer con atención bidireccional para comprensión", "Un tipo de red recurrente mejorada", "Un buscador de sinónimos"],
        "correcta_id": 1,
        "explicacion": "BERT está diseñado para entender el contexto de forma bidireccional profunda, ideal para tareas de clasificación y extracción de información."
    },
    {
        "pregunta": "¿En qué consiste el pre-entrenamiento Masked LM de BERT?",
        "opciones": ["En traducir textos al azar", "En ocultar palabras de la oración para que el modelo intente adivinarlas usando el contexto", "En borrar oraciones completas", "En cambiar el orden de las letras"],
        "correcta_id": 1,
        "explicacion": "Permite que el modelo aprenda representaciones bidireccionales al tener que mirar a ambos lados para predecir la palabra faltante."
    },
    {
        "pregunta": "¿Qué es GPT?",
        "opciones": ["Un modelo de comprensión bidireccional", "Un modelo de lenguaje generativo autoregresivo basado en el Decoder del Transformer", "Un algoritmo de compresión de datos", "Un sistema de reglas gramaticales"],
        "correcta_id": 1,
        "explicacion": "GPT se especializa en la generación de texto prediciendo la siguiente palabra de forma unidireccional."
    },
    {
        "pregunta": "¿Cuál es la diferencia fundamental entre el entrenamiento de GPT y BERT?",
        "opciones": ["GPT usa imágenes y BERT no", "BERT es bidireccional (comprensión) y GPT es unidireccional/causal (generación)", "BERT es más rápido que GPT", "GPT no usa Transformers"],
        "correcta_id": 1,
        "explicacion": "BERT mira toda la oración a la vez; GPT solo mira hacia atrás para poder generar el futuro de forma coherente."
    },
    {
        "pregunta": "¿Qué significa que un modelo es autoregresivo?",
        "opciones": ["Que retrocede en el tiempo", "Que usa sus propias salidas previas como entradas para el siguiente paso", "Que tiene pocos parámetros", "Que no puede aprender de los datos"],
        "correcta_id": 1,
        "explicacion": "Es la base de la generación de texto: cada palabra generada se suma al contexto para decidir la siguiente."
    },
    {
        "pregunta": "¿Qué rol juega el token [CLS] en BERT?",
        "opciones": ["Representar el final de una oración", "Servir como una representación agregada de toda la secuencia para tareas de clasificación", "Indicar una palabra fuera de vocabulario", "Separar dos oraciones distintas"],
        "correcta_id": 1,
        "explicacion": "El token [CLS] se coloca al inicio y su vector de salida se utiliza como el 'resumen' semántico de la secuencia completa para alimentar un clasificador final [Fuente 16]."
    },
    {
        "pregunta": "¿Qué problemas de escalabilidad introduce la auto-atención (self-attention) al crecer la secuencia?",
        "opciones": ["Su complejidad es lineal O(N)", "Su complejidad es cuadrática O(N²), lo que dispara el uso de memoria y cómputo", "No tiene problemas de escalabilidad", "Hace que el modelo sea más pequeño"],
        "correcta_id": 1,
        "explicacion": "Cada token debe atender a todos los demás, por lo que duplicar el largo de la secuencia cuadriplica el costo computacional [Fuente 16, 20]."
    },
    {
        "pregunta": "¿Aprenden lo mismo todas las cabezas de atención en un Transformer?",
        "opciones": ["Sí, son copias idénticas para redundancia", "No, cada cabeza puede especializarse en capturar relaciones distintas (sintácticas, semánticas o de larga distancia)", "Solo aprenden el orden de las palabras", "Se usan para procesar distintos idiomas"],
        "correcta_id": 1,
        "explicacion": "La atención multi-cabeza permite al modelo atender simultáneamente a información de diferentes subespacios de representación [Fuente 17]."
    },
    {
        "pregunta": "¿Cómo representarías documentos más largos que el límite máximo de BERT (512 tokens)?",
        "opciones": ["Ignorando el resto del texto", "Dividiendo el documento en fragmentos (chunks) y promediando o aplicando estrategias de jerarquía", "Aumentando el embedding size manualmente", "No se puede hacer"],
        "correcta_id": 1,
        "explicacion": "Debido a la restricción cuadrática, para textos largos se suelen procesar ventanas solapadas o utilizar modelos diseñados para secuencias largas como Longformer [Fuente 17]."
    },
    {
        "pregunta": "¿En qué capas se centra principalmente el Pre-entrenamiento (PT) y en cuáles el Fine-tuning (FT)?",
        "opciones": ["PT en las últimas y FT en las primeras", "PT en todas las capas para aprender el lenguaje y FT suele centrarse en las capas superiores o cabezas específicas de tarea", "PT solo en los embeddings", "Son exactamente iguales"],
        "correcta_id": 1,
        "explicacion": "El pre-entrenamiento busca un conocimiento general, mientras que el ajuste fino adapta las capas finales para la clasificación o tarea específica [Fuente 20]."
    },
    {
        "pregunta": "¿Qué son los Adapters en el contexto de Fine-tuning?",
        "opciones": ["Cables para conectar GPUs", "Pequeños módulos entrenables que se insertan entre las capas fijas del modelo pre-entrenado", "Algoritmos de limpieza de texto", "Un tipo de tokenizador"],
        "correcta_id": 1,
        "explicacion": "Permiten un 'Parameter-Efficient Fine-Tuning', donde solo se entrenan unos pocos parámetros nuevos manteniendo el modelo original congelado [Fuente 20]."
    },
    {
        "pregunta": "¿Qué es LoRA (Low-Rank Adaptation)?",
        "opciones": ["Un método de pre-entrenamiento rápido", "Una técnica que inyecta matrices de bajo rango entrenables en las capas de atención para un fine-tuning eficiente", "Un modelo competidor de GPT-4", "Una forma de compresión de archivos"],
        "correcta_id": 1,
        "explicacion": "LoRA reduce drásticamente la cantidad de parámetros a entrenar al optimizar solo las actualizaciones de rango bajo de las matrices de pesos [Fuente 20]."
    },
    {
        "pregunta": "¿Qué son las capacidades emergentes en los LLMs?",
        "opciones": ["Errores que aparecen al azar", "Habilidades (como razonamiento lógico o traducción) que aparecen solo cuando el modelo alcanza un tamaño crítico de parámetros y datos", "Funciones programadas explícitamente por humanos", "Capacidades de ahorro de energía"],
        "correcta_id": 1,
        "explicacion": "Son comportamientos que no están presentes en modelos pequeños pero se manifiestan de forma abrupta al escalar la computación [Fuente 20]."
    },
    {
        "pregunta": "¿Cuál es la diferencia principal entre un modelo Base y uno Instruction Tuned?",
        "opciones": ["El modelo Base es más inteligente", "El modelo Base completa texto estadísticamente; el Instruction Tuned está alineado para seguir órdenes del usuario", "El Instruction Tuned no usa Transformers", "No hay diferencia en el comportamiento"],
        "correcta_id": 1,
        "explicacion": "Un modelo base puede completar una pregunta con otra pregunta; el instruction tuned responderá a la orden gracias al entrenamiento supervisado [Fuente 18, 20]."
    },
    {
        "pregunta": "¿Cuáles son los tres pasos estándar para generar un asistente como ChatGPT?",
        "opciones": ["1. Limpieza, 2. Tokenización, 3. Inferencia", "1. Pre-entrenamiento, 2. SFT (Supervised Fine-Tuning), 3. RLHF (Refuerzo por feedback humano)", "1. Leer libros, 2. Hablar con gente, 3. Guardar en disco", "No requiere pasos, aprende solo"],
        "correcta_id": 1,
        "explicacion": "Primero se aprende el lenguaje (PT), luego se aprende a seguir instrucciones (SFT) y finalmente se alinea con preferencias humanas (RLHF) [Fuente 18, 20]."
    },
    {
        "pregunta": "¿Qué es el In-context Learning?",
        "opciones": ["Entrenar el modelo con nuevos datos de contexto", "La capacidad del modelo de aprender una tarea a partir de ejemplos dados en el prompt sin actualizar sus pesos", "Un tipo de memoria RAM", "Guardar el historial de chat en una base de datos"],
        "correcta_id": 1,
        "explicacion": "Permite que el modelo entienda qué queremos que haga simplemente dándole unos pocos ejemplos (few-shot) en el texto de entrada [Fuente 20]."
    },
    {
        "pregunta": "¿Qué son las alucinaciones en un LLM?",
        "opciones": ["Cuando el modelo se apaga solo", "Generación de información que suena convincente pero es falsa o no está sustentada en los datos", "Un modo de ahorro de energía", "Cuando el modelo usa muchos tokens"],
        "correcta_id": 1,
        "explicacion": "Ocurren porque el modelo es un predictor probabilístico de tokens y no una base de datos de hechos verídicos [Fuente 20]."
    },
    {
        "pregunta": "¿Cómo se pueden mitigar las alucinaciones?",
        "opciones": ["Pidiendo al modelo que 'piense paso a paso' o usando técnicas de RAG (Retrieval Augmented Generation)", "Apagando el modelo por las noches", "Usando un vocabulario más pequeño", "No se pueden mitigar"],
        "correcta_id": 0,
        "explicacion": "Proveer fuentes externas (RAG) o forzar el razonamiento explícito ayuda a reducir la invención de hechos [Fuente 2, 20]."
    },
    {
        "pregunta": "¿Qué tipo de prompts funcionan mejor en modelos Base?",
        "opciones": ["Órdenes directas como 'Resumí este texto'", "Prompts de completitud (Few-shot) donde se le muestra el patrón esperado", "Preguntas cortas de sí o no", "Cualquier prompt funciona igual"],
        "correcta_id": 1,
        "explicacion": "Los modelos base son 'completadores' de texto; imitan el patrón que ven. Por eso los ejemplos (few-shot) son vitales para ellos [Fuente 18, 20]."
    },
    {
        "pregunta": "¿Se pueden recolectar datos para Instruction Tuning de forma sintética?",
        "opciones": ["No, solo pueden ser escritos por humanos", "Sí, usando un modelo muy potente (ej. GPT-4) para generar pares de instrucción-respuesta para entrenar modelos más chicos", "Solo se pueden sacar de Wikipedia", "Es ilegal"],
        "correcta_id": 1,
        "explicacion": "Técnicas como las usadas en modelos como Alpaca o Vicuna utilizan destilación de conocimiento de modelos más grandes [Fuente 19, 20]."
    },
    {
        "pregunta": "¿Qué es el RLHF (Reinforcement Learning from Human Feedback)?",
        "opciones": ["Un método para pre-entrenar modelos", "Un proceso de optimización donde el modelo aprende de las preferencias y correcciones de evaluadores humanos", "Una técnica de tokenización", "Un sistema de seguridad de hardware"],
        "correcta_id": 1,
        "explicacion": "Se utiliza para alinear los LLMs con valores humanos de utilidad, honestidad y seguridad [Fuente 18, 20]."
    },
    {
        "pregunta": "¿Cuál es el objetivo de entrenar un Reward Model (RM) en RLHF?",
        "opciones": ["Predecir la siguiente palabra de un texto", "Puntuar las respuestas del modelo según las preferencias humanas", "Traducir textos entre idiomas", "Reducir el tamaño del vocabulario"],
        "correcta_id": 1,
        "explicacion": "El RM actúa como un 'juez' matemático que aprende qué respuestas prefiere un humano, permitiendo luego optimizar al LLM mediante aprendizaje por refuerzo [Fuente 1, 7]."
    },
    {
        "pregunta": "¿Qué tipo de datos se necesitan para entrenar un Reward Model?",
        "opciones": ["Solo texto de Wikipedia", "Pares de respuestas donde un humano indica cuál es mejor (A > B)", "Etiquetas de sentimientos positivo/negativo", "Código de programación"],
        "correcta_id": 1,
        "explicacion": "Se utiliza el 'Bradley-Terry model' sobre comparaciones binarias para que el modelo aprenda un ranking de calidad basado en el feedback humano [Fuente 7]."
    },
    {
        "pregunta": "¿Qué ventaja tiene DPO (Direct Preference Optimization) frente a PPO/RLHF?",
        "opciones": ["Es más lento pero más preciso", "Elimina la necesidad de entrenar y mantener un Reward Model por separado", "Solo funciona con modelos pequeños", "Requiere más memoria GPU"],
        "correcta_id": 1,
        "explicacion": "DPO simplifica el proceso al derivar una solución cerrada que optimiza la política directamente sobre los datos de preferencia, evitando la inestabilidad de los algoritmos de RL tradicionales [Fuente 1]."
    },
    {
        "pregunta": "¿Qué es el Chain-of-Thought (CoT) prompting?",
        "opciones": ["Generar texto en varios idiomas a la vez", "Pedir al modelo que explique su razonamiento paso a paso antes de dar la respuesta final", "Una técnica para comprimir el modelo", "Un método de tokenización por caracteres"],
        "correcta_id": 1,
        "explicacion": "Forzar al modelo a 'pensar' en voz alta mejora drásticamente su rendimiento en tareas de lógica, matemáticas y razonamiento complejo [Fuente 1]."
    },
    {
        "pregunta": "¿Cuál es la innovación clave de modelos como DeepSeek-R1 o modelos de razonamiento (Reasoning Models)?",
        "opciones": ["Son más pequeños que el promedio", "Utilizan RL para incentivar el pensamiento previo (thinking) de forma nativa", "No usan mecanismos de atención", "Solo sirven para búsqueda de información"],
        "correcta_id": 1,
        "explicacion": "Estos modelos están entrenados específicamente para generar una cadena de pensamiento interna antes de responder, optimizada mediante procesos de refuerzo como GRPO [Fuente 1]."
    },
    {
        "pregunta": "¿Qué significa RAG (Retrieval-Augmented Generation)?",
        "opciones": ["Entrenar el modelo con más datos", "Aumentar la generación de texto con información recuperada de una fuente externa en tiempo real", "Reducir el sesgo del modelo", "Un algoritmo de limpieza de datos"],
        "correcta_id": 1,
        "explicacion": "RAG combina la capacidad de redacción del LLM con la precisión de una base de datos externa, mitigando alucinaciones y permitiendo usar datos privados o actualizados [Fuente 2]."
    },
    {
        "pregunta": "¿Qué componente de RAG se encarga de convertir documentos en vectores?",
        "opciones": ["El generador", "El retriever (modelo de embeddings)", "La función Softmax", "El tokenizador BPE"],
        "correcta_id": 1,
        "explicacion": "El retriever utiliza modelos de dense retrieval para mapear la consulta y los documentos a un espacio vectorial donde pueda buscar similitudes [Fuente 2]."
    },
    {
        "pregunta": "¿Qué es un Agente de IA en el contexto moderno?",
        "opciones": ["Un chatbot que solo responde preguntas", "Un sistema capaz de planificar, usar herramientas externas (browsing, código) y ejecutar bucles de razonamiento para cumplir un objetivo", "Un modelo pre-entrenado sin fine-tuning", "Un programa que solo clasifica correos"],
        "correcta_id": 1,
        "explicacion": "Los agentes utilizan al LLM como 'cerebro' para decidir qué herramientas usar y cómo iterar hasta resolver una tarea compleja [Fuente 2]."
    },
    {
        "pregunta": "¿Qué es la destilación de conocimiento (Knowledge Distillation)?",
        "opciones": ["Borrar datos del modelo", "Entrenar un modelo pequeño (estudiante) para que imite el comportamiento de un modelo grande (maestro)", "Fusionar dos modelos distintos", "Aumentar el número de capas"],
        "correcta_id": 1,
        "explicacion": "Permite transferir la capacidad de modelos gigantes a otros más eficientes y rápidos de ejecutar en dispositivos con pocos recursos [Fuente 2]."
    },
    {
        "pregunta": "Si duplicás el largo de contexto de un modelo (e.g., de 4K a 8K tokens), ¿qué sucede?",
        "opciones": ["La memoria del KV cache se duplica y el costo de la atención se cuadruplica", "Todo se mantiene igual", "La memoria se reduce a la mitad", "El modelo se vuelve lineal"],
        "correcta_id": 0,
        "explicacion": "El KV cache escala de forma lineal O(N) con la secuencia, pero el mecanismo de atención original tiene una complejidad cuadrática O(N²) [Fuente 11]."
    },
    {
        "pregunta": "¿Por qué modelos como LLaMA dividen los números en dígitos individuales durante la tokenización?",
        "opciones": ["Para ahorrar espacio", "Para mejorar la capacidad del modelo en tareas aritméticas y evitar sesgos de frecuencia de números específicos", "Porque es más fácil de programar", "Para que el modelo lea más rápido"],
        "correcta_id": 1,
        "explicacion": "Al ver cada dígito por separado, el modelo puede aprender reglas matemáticas generales en lugar de tratar cada número compuesto como un símbolo único y raro [Fuente 11]."
    },
    {
        "pregunta": "¿Qué factores técnicos limitan el tamaño máximo de la ventana de contexto?",
        "opciones": ["Solo la velocidad de internet", "La memoria VRAM disponible (KV cache) y la capacidad del positional encoding (ej. límites de RoPE)", "La cantidad de palabras en el diccionario", "No hay límites técnicos"],
        "correcta_id": 1,
        "explicacion": "Secuencias muy largas requieren gigabytes de VRAM solo para el cache y técnicas especiales para que el modelo entienda posiciones que no vio en el entrenamiento [Fuente 11]."
    },
    {
        "pregunta": "¿Aproximadamente qué porcentaje de los parámetros de un Transformer decoder-only suele estar en las FFN?",
        "opciones": ["El 10%", "Aproximadamente 2/3 (66%) de los parámetros totales", "Casi el 100%", "Están repartidos 50/50 con la atención"],
        "correcta_id": 1,
        "explicacion": "Aunque conceptualmente simples, las capas Feed-Forward suelen expandir la dimensión del modelo (típicamente 4x), acumulando la mayor parte del conocimiento estático [Fuente 11]."
    },
    {
        "pregunta": "¿Cuál es el beneficio principal de GQA (Grouped Query Attention)?",
        "opciones": ["Hace que el modelo sea más creativo", "Reduce significativamente el tamaño del KV cache y acelera la inferencia sin perder mucha calidad", "Permite usar un vocabulario más grande", "Elimina la necesidad de embeddings"],
        "correcta_id": 1,
        "explicacion": "Al compartir 'Keys' y 'Values' entre varios 'Queries', se reduce drásticamente el tráfico de memoria durante la generación de texto [Fuente 1, 11]."
    },
    {
        "pregunta": "En un modelo de 32 capas con d=4096 y vocabulario de 32K, ¿qué componente consume más memoria en los pesos iniciales?",
        "opciones": ["El embedding de entrada", "Las capas de atención y FFN combinadas", "Las Layer Norms", "El positional encoding"],
        "correcta_id": 1,
        "explicacion": "En modelos de gran escala, los pesos de las 32 capas (bloques del Transformer) superan por mucho al embedding de vocabulario [Fuente 11]."
    }
]

# ── Perfiles de usuario ──────────────────────────────────────────────────────
# En Render el filesystem es efímero: sin disco persistente, los datos se pierden
# al redeploy/reinicio. Opciones: variable USUARIOS_JSON_PATH apuntando a un
# volumen montado (Render Disk) o una base externa (p. ej. Postgres).

def _ruta_usuarios_json() -> str:
    custom = (os.environ.get("USUARIOS_JSON_PATH") or "").strip()
    if custom:
        return os.path.abspath(custom)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "usuarios.json"))

def _cargar_perfiles() -> dict:
    ruta = _ruta_usuarios_json()
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _guardar_perfiles(perfiles: dict):
    ruta = _ruta_usuarios_json()
    parent = os.path.dirname(ruta)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(perfiles, f, ensure_ascii=False, indent=2)

perfiles_usuarios = _cargar_perfiles()

def registrar_usuario(user):
    """Crea o actualiza el perfil de un usuario de Telegram."""
    uid = str(user.id)
    if uid not in perfiles_usuarios:
        perfiles_usuarios[uid] = {
            "id": user.id,
            "nombre": user.first_name,
            "apellido": user.last_name or "",
            "username": user.username or "",
            "total": 0,
            "correctas": 0,
            "incorrectas": 0,
            "primera_vez": time.strftime("%Y-%m-%d %H:%M"),
            "ultima_actividad": time.strftime("%Y-%m-%d %H:%M"),
        }
    else:
        perfiles_usuarios[uid]["nombre"] = user.first_name
        perfiles_usuarios[uid]["apellido"] = user.last_name or ""
        perfiles_usuarios[uid]["username"] = user.username or ""
        perfiles_usuarios[uid]["ultima_actividad"] = time.strftime("%Y-%m-%d %H:%M")
    _guardar_perfiles(perfiles_usuarios)

def registrar_respuesta(user_id: int, acerto: bool):
    """Suma una respuesta al perfil del usuario."""
    uid = str(user_id)
    if uid not in perfiles_usuarios:
        return
    perfiles_usuarios[uid]["total"] += 1
    if acerto:
        perfiles_usuarios[uid]["correctas"] += 1
    else:
        perfiles_usuarios[uid]["incorrectas"] += 1
    perfiles_usuarios[uid]["ultima_actividad"] = time.strftime("%Y-%m-%d %H:%M")
    _guardar_perfiles(perfiles_usuarios)

# ── Colas y encuestas activas ────────────────────────────────────────────────

# Guarda una cola de preguntas por usuario: { user_id: [lista_mezclada] }
colas_usuarios = {}

# Mapea poll_id → datos de la pregunta para poder mostrar la teoría al responder
encuestas_activas = {}

# Guarda la última pregunta respondida por usuario (para el botón "Más teoría")
ultima_pregunta_usuario = {}

def obtener_siguiente_pregunta(user_id):
    """Devuelve la próxima pregunta sin repetir. Cuando se acaban, reinicia el ciclo."""
    if user_id not in colas_usuarios or len(colas_usuarios[user_id]) == 0:
        cola = preguntas.copy()
        random.shuffle(cola)
        colas_usuarios[user_id] = cola
    return colas_usuarios[user_id].pop()

def truncar(texto, limite=100):
    """Trunca el texto al límite de Telegram dejando '…' al final si es necesario."""
    return texto if len(texto) <= limite else texto[:limite - 1] + "…"

def enviar_pregunta_privada(chat_id):
    q = mejorar_pregunta(obtener_siguiente_pregunta(chat_id))

    # Barajamos los índices (no los textos) para evitar problemas con textos duplicados o truncados
    indices = list(range(len(q["opciones"])))
    random.shuffle(indices)

    opciones_enviadas = [truncar(q["opciones"][i]) for i in indices]
    pregunta_segura = truncar(q["pregunta"], 300)

    # El índice correcto en la lista enviada a Telegram
    nuevo_correcta_id = indices.index(q["correcta_id"])

    mensaje = bot.send_poll(
        chat_id=chat_id,
        question=pregunta_segura,
        options=opciones_enviadas,
        type="quiz",
        correct_option_id=nuevo_correcta_id,
        explanation=q["explicacion"],
        is_anonymous=False
    )
    # Guardamos pregunta + índice correcto tal como fue enviado a Telegram
    encuestas_activas[mensaje.poll.id] = {**q, "_correcta_enviada": nuevo_correcta_id}

# ── Handlers para conversación 1:1 ──────────────────────────────────────────

@bot.message_handler(commands=["start", "empezar"])
def cmd_start(message):
    registrar_usuario(message.from_user)
    bot.reply_to(message, f"¡Hola, {message.from_user.first_name}! Soy el bot de repaso de MIA. Voy a enviarte preguntas de a una. ¡Respondé cada encuesta para recibir la siguiente!\n\nUsá /stats para ver tu progreso.")
    enviar_pregunta_privada(message.chat.id)

@bot.message_handler(commands=["stats"])
def cmd_stats(message):
    registrar_usuario(message.from_user)
    uid = str(message.from_user.id)
    p = perfiles_usuarios.get(uid)
    if not p or p["total"] == 0:
        bot.reply_to(message, "Todavía no respondiste ninguna pregunta. ¡Usá /empezar para arrancar!")
        return
    pct = round(p["correctas"] / p["total"] * 100)
    texto = (
        f"📊 *Tu progreso, {p['nombre']}*\n\n"
        f"✅ Correctas: {p['correctas']}\n"
        f"❌ Incorrectas: {p['incorrectas']}\n"
        f"📝 Total respondidas: {p['total']}\n"
        f"🎯 Acierto: {pct}%\n\n"
        f"🕐 Última actividad: {p['ultima_actividad']}"
    )
    bot.reply_to(message, texto, parse_mode="Markdown")

def responder_pregunta_libre(pregunta: str) -> str:
    """Responde cualquier pregunta del usuario usando el contexto de la clase."""
    prompt = f"""Sos un tutor experto del curso MIA305 - NLP (Maestría en IA Aplicada). Un alumno te hace la siguiente pregunta:

"{pregunta}"

Usá el siguiente material de clase como referencia principal:
---
{CONTEXTO_CLASE}
---

Respondé de forma clara, directa y pedagógica (máximo 5 oraciones). Si el tema involucra matemática, usá notación simple:
- Fracciones: a/b  |  Raíces: √x  |  Potencias: x²  |  Vectores: v ∈ ℝ^d
- PROHIBIDO LaTeX: no uses \\frac, \\sqrt, $...$, \\mathbf ni similares.

Si la pregunta no está relacionada con el curso, indicalo amablemente y redirigí al alumno.
Sin saludos ni introducciones. Solo la respuesta."""

    try:
        respuesta = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] Error en pregunta libre: {e}")
        return "No pude procesar tu pregunta en este momento. Intentá de nuevo en unos segundos."

def teclado_volver_quiz():
    teclado = telebot.types.InlineKeyboardMarkup()
    teclado.add(telebot.types.InlineKeyboardButton("🎯 Volver al quiz", callback_data="siguiente_pregunta"))
    return teclado

@bot.message_handler(func=lambda m: True)
def mensaje_generico(message):
    registrar_usuario(message.from_user)
    bot.send_chat_action(message.chat.id, "typing")
    respuesta = responder_pregunta_libre(message.text)
    bot.reply_to(message, respuesta, parse_mode="Markdown", reply_markup=teclado_volver_quiz())

def teclado_post_respuesta():
    """Crea el teclado inline con las dos acciones disponibles tras responder."""
    teclado = telebot.types.InlineKeyboardMarkup()
    teclado.row(
        telebot.types.InlineKeyboardButton("📚 Más teoría", callback_data="mas_teoria"),
        telebot.types.InlineKeyboardButton("➡️ Siguiente pregunta", callback_data="siguiente_pregunta"),
    )
    return teclado

@bot.poll_answer_handler()
def al_responder(poll_answer):
    """Se dispara cada vez que un usuario vota en una encuesta privada."""
    user_id = poll_answer.user.id
    poll_id = poll_answer.poll_id
    opcion_elegida = poll_answer.option_ids[0] if poll_answer.option_ids else None

    q = encuestas_activas.pop(poll_id, None)

    if q and opcion_elegida is not None:
        acerto = opcion_elegida == q["_correcta_enviada"]
        respuesta_correcta = q["opciones"][q["correcta_id"]]

        registrar_respuesta(user_id, acerto)
        ultima_pregunta_usuario[user_id] = q  # guardamos para "Más teoría"

        if acerto:
            encabezado = "✅ ¡Correcto!"
        else:
            encabezado = f"❌ Incorrecto. La respuesta era: *{respuesta_correcta}*"

        explicacion = generar_explicacion(q, acerto)
        texto = f"{encabezado}\n\n📖 *Explicación:*\n{explicacion}"
        bot.send_message(user_id, texto, parse_mode="Markdown",
                         reply_markup=teclado_post_respuesta())
    else:
        # Si no hay contexto (caso raro), mandamos directamente la siguiente
        enviar_pregunta_privada(user_id)

@bot.callback_query_handler(func=lambda call: call.data == "siguiente_pregunta")
def cb_siguiente(call):
    bot.answer_callback_query(call.id)
    # Eliminamos los botones del mensaje anterior para no acumular teclados
    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
    enviar_pregunta_privada(call.from_user.id)

@bot.callback_query_handler(func=lambda call: call.data == "mas_teoria")
def cb_mas_teoria(call):
    bot.answer_callback_query(call.id, text="Generando explicación profunda…")
    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

    q = ultima_pregunta_usuario.get(call.from_user.id)
    if not q:
        bot.send_message(call.from_user.id, "No encontré la pregunta anterior. Usá /empezar para continuar.")
        return

    teoria = generar_teoria_profunda(q)
    bot.send_message(
        call.from_user.id,
        f"🔬 *Teoría profunda:*\n\n{teoria}",
        parse_mode="Markdown",
        reply_markup=telebot.types.InlineKeyboardMarkup().row(
            telebot.types.InlineKeyboardButton("➡️ Siguiente pregunta", callback_data="siguiente_pregunta")
        )
    )


def _webhook_path_secret() -> str:
    """Segmento de URL no adivinable; Telegram solo debe POSTear aquí."""
    explicit = (os.environ.get("WEBHOOK_SECRET") or "").strip()
    if explicit:
        return explicit
    t = os.getenv("token") or ""
    if len(t) < 16:
        return "cambiar-webhook-secret"
    return hashlib.sha256(t.encode()).hexdigest()[:32]


def create_web_app():
    """
    Servidor HTTP para Render (Web Service): salud en / y webhook de Telegram.
    Con RENDER_EXTERNAL_URL (lo define Render) registra webhook para que cada
    mensaje despierte la instancia; sin eso, hace polling en un hilo (útil en local).
    """
    from flask import Flask, request

    app = Flask(__name__)
    secret = _webhook_path_secret()

    @app.route(f"/tgwh/{secret}", methods=["POST"])
    def telegram_webhook():
        if not request.is_json:
            return "", 400
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return "", 400
        try:
            update = telebot.types.Update.de_json(body)
            bot.process_new_updates([update])
        except Exception as e:
            print(f"[webhook] error al procesar update: {e}")
            return "", 500
        return "", 200

    @app.get("/")
    def health():
        return "ok", 200

    base = (os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
    if base and TOKEN:
        hook = f"{base}/tgwh/{secret}"
        try:
            bot.remove_webhook()
            time.sleep(0.25)
            bot.set_webhook(
                url=hook,
                max_connections=40,
                allowed_updates=[
                    "message",
                    "edited_message",
                    "callback_query",
                    "poll_answer",
                ],
            )
            print(f"[webhook] registrado: {hook}")
        except Exception as e:
            print(f"[webhook] no se pudo registrar ({e}); usando polling en segundo plano.")
            threading.Thread(
                target=lambda: bot.infinity_polling(timeout=60, long_polling_timeout=60),
                daemon=True,
            ).start()
    else:
        print("[web] Sin RENDER_EXTERNAL_URL o token vacío: polling en segundo plano.")
        threading.Thread(
            target=lambda: bot.infinity_polling(timeout=60, long_polling_timeout=60),
            daemon=True,
        ).start()

    return app


web_app = None
if os.environ.get("PORT"):
    web_app = create_web_app()

if __name__ == "__main__":
    port = os.environ.get("PORT")
    if port and web_app is not None:
        web_app.run(host="0.0.0.0", port=int(port), threaded=True)
    else:
        print("Bot iniciado. Esperando mensajes (polling local)...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)