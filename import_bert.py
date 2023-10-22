import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

def import_bert(x, text_column, model_name="bert-base-multilingual-cased", layer_index=12, batch_size=2,token_index_filter=1):
    """
    x: pd.DataFrame, the input data.
    text_column: str, the name of the column containing the text data.
    model_name: str, the name of the BERT model to use.
    layer_index: int, the index of the BERT layer to extract embeddings from.
    batch_size: int, the size of batches to process at a time.
    """
    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load pre-trained model
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    
    # Ensure model is in evaluation mode (not training mode)
    model.eval()
    
    # Initialize list to store embeddings
    embeddings = []
    
    # Process text data in batches
    for i in range(0, len(x[text_column]), batch_size):
        # Tokenize text
        # print(x[text_column][i:i+batch_size].tolist())
        inputs = tokenizer(
            x[text_column][i:i+batch_size].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract embeddings from the desired layer
        hidden_states = outputs.hidden_states[layer_index]
        token_embeddings = hidden_states[:, token_index_filter, :]

        # Append embeddings to list
        embeddings.extend(token_embeddings.numpy())
    
    # Convert list of embeddings to DataFrame and return
    return pd.DataFrame(embeddings)