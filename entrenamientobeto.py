# ============================================================================
# Nombre del archivo: multiplestxt_to_asingletxt.py
# Autor: Jonathan Zavala-Díaz
# Fecha: 26 de Junio de 2023
# Descripción: Agrupar multiples archivos de texto en uno solo
# ============================================================================

from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
import torch
import os
import time
import random
from torch.optim import AdamW
def mask_randomly(text, tokenizer, mask_probability=0.15):
    tokenized_text = tokenizer.tokenize(text)
    masked_tokens = []
    for token in tokenized_text:
        if random.random() < mask_probability:
            # Con una probabilidad de 15%, reemplazamos el token por [MASK]
            masked_tokens.append(tokenizer.mask_token)
        else:
            masked_tokens.append(token)
    return tokenizer.convert_tokens_to_string(masked_tokens)

start_time = time.time()
# Tokenizador y modelo preentrenado Beto
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Directorio que contiene tus archivos de texto con notas clínicas
data_directory = "data/entrenamientobeto"

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
encodings = tokenizer(clinical_notes, padding=True, truncation=True, return_tensors="pt")

# Crear un conjunto de datos PyTorch
class FillMaskDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['attention_mask'] = torch.tensor([1 if i != 0 else 0 for i in item['input_ids']])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = FillMaskDataset(encodings)

# Configuración del entrenamiento
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Entrenamiento
model.train()
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Durante el entrenamiento, asegúrate de pasar la máscara de atención junto con tus input_ids
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  # Añade la máscara de atención aquí
        labels = input_ids.clone()
        labels[labels != tokenizer.mask_token_id] = -100
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Pasa la máscara de atención aquí
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Guardar el modelo entrenado
model.save_pretrained("modelo/modelo_entrenado")
tokenizer.save_pretrained("modelo/tokenizer")
# Medir el tiempo de llenado de espacios en blanco

execution_time = time.time() - start_time

print(f"Tiempo de ejecución: {execution_time:.2f} segundos")