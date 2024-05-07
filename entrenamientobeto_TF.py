# ============================================================================
# Nombre del archivo: multiplestxt_to_asingletxt.py
# Autor: Jonathan Zavala-Díaz
# Fecha: 26 de Junio de 2023
# Descripción: Agrupar multiples archivos de texto en uno solo
# ============================================================================
from transformers import BertTokenizer, TFBertForMaskedLM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset
import tensorflow as tf
import os
import time
import random

def mask_randomly(text, tokenizer, mask_probability=0.15):
    tokenized_text = tokenizer.tokenize(text)
    masked_tokens = []
    for token in tokenized_text:
        if random.random() < mask_probability:
            masked_tokens.append(tokenizer.mask_token)
        else:
            masked_tokens.append(token)
    return tokenizer.convert_tokens_to_string(masked_tokens)

start_time = time.time()

# Tokenizador y modelo preentrenado Beto
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = TFBertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Directorio que contiene tus archivos de texto con notas clínicas
data_directory = "data"

# Lista para almacenar los textos de tus notas clínicas
clinical_notes = []

# Leer archivos de texto, enmascarar y agregar los textos a la lista
for filename in os.listdir(data_directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                # Aplicar enmascaramiento aleatorio
                masked_text = mask_randomly(text, tokenizer)
                clinical_notes.append(masked_text)
        except Exception as e:
            print(f"Error al leer el archivo {filename}: {e}")
            continue

# Tokenización de las notas clínicas
encodings = tokenizer(clinical_notes, padding=True, truncation=True, return_tensors="tf")

# Crear un conjunto de datos TensorFlow
class FillMaskDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['attention_mask'] = tf.where(item['input_ids'] != 0, 1, 0)
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = FillMaskDataset(encodings)

# Configuración del entrenamiento
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Entrenamiento
train_loader = tf.data.Dataset.from_tensor_slices(train_dataset).batch(2)

# Durante el entrenamiento, asegúrate de pasar la máscara de atención junto con tus input_ids
for epoch in range(3):
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]  # Añade la máscara de atención aquí
        labels = tf.where(input_ids != tokenizer.mask_token_id, input_ids, -100)
        with tf.GradientTape() as tape:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Pasa la máscara de atención aquí
            loss = outputs.loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Guardar el modelo entrenado
model.save_pretrained("modelo/modelo_entrenado")
tokenizer.save_pretrained("modelo/tokenizer")

# Medir el tiempo de ejecución
execution_time = time.time() - start_time

print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
