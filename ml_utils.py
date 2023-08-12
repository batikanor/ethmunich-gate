# from transformers import BertTokenizer, BertModel
# import torch

# # Load pre-trained model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Encode text
# text = "Embeddings are useful for NLP tasks."
# encoded_input = tokenizer(text, return_tensors='pt')

# # Get embeddings
# with torch.no_grad():
#     output = model(**encoded_input)
    
# # The last hidden state is the sequence of hidden states of the last layer of the model.
# embeddings = output.last_hidden_state

# # If you want the embedding for [CLS] token (often used for classification tasks)
# cls_embedding = embeddings[0][0]

# print(cls_embedding)
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Load pretrained ResNet50 model + higher level layers
resnetmodel = models.resnet50(pretrained=True)
resnetmodel = torch.nn.Sequential(*(list(resnetmodel.children())[:-1]))  # Remove the last layer
resnetmodel.eval()

# Define image transformations - normalization and resizing as per model's requirements
resnettransformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_embedding(pil_img):
    # Open and preprocess the image
    image = pil_img
    image = resnettransformer(image).unsqueeze(0)  # Transform the image and add batch dimension

    # Extract the embedding
    with torch.no_grad():
        embedding = resnetmodel(image)
    return embedding.squeeze().numpy()  # Convert the tensor to numpy array


# Initialize the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# def get_embedding(text):
#     encoded_input = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         output = model(**encoded_input)
#     # Get the [CLS] token embedding and squeeze to make it 1D
#     return output.last_hidden_state[:, 0, :].squeeze()

# def get_embedding(text):
#     # Tokenize the text and truncate to fit within BERT's limit
#     tokens = tokenizer.tokenize(text)
#     tokens = tokens[:512 - 2]  # Account for [CLS] and [SEP] tokens
#     encoded_input = tokenizer.encode_plus(tokens, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#     with torch.no_grad():
#         output = model(**encoded_input['input_ids'])
#     return output.last_hidden_state[:, 0, :].squeeze()




def reduce_dimensions(embeddings, method='PCA', n_components=2):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Method can be either 'PCA' or 'TSNE'")
    
    return reducer.fit_transform(embeddings)

def get_embedding(text):
    encoded_input = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        output = model(**encoded_input)  # Use the entire dictionary, not just input_ids
    return output.last_hidden_state[:, 0, :].squeeze()

def cosine_similarity(embedding1, embedding2):
    # Compute the dot product
    dot_product = torch.dot(embedding1, embedding2)
    # Compute the L2 norms (euclidean norms)
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    # Cosine similarity formula
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()


def main():
    text1 = "The sky is blue."
    text2 = "The ocean is vast."

    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    similarity_score = cosine_similarity(embedding1, embedding2)
    print(f"Cosine Similarity: {similarity_score}")
