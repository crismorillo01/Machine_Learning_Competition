import os
import json
from PIL import Image
import requests
import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def submit(results, groupname, url):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


def batching(images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        tmp_images = images[i:i + batch_size]
        inputs = torch.stack([preprocess(img.convert("RGB"))
                             for img in tmp_images]).to(device)
        with torch.no_grad():
            tmp_features = model(inputs)
            features.append(tmp_features)
    return torch.cat(features, dim=0)


data_folder = "/Users/cristinamorilloleal/Documents/Máster Data Science/Primer Curso/Segundo Cuatri/Introduction to ML/Competition Project/test_data"
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

query_images = []
query_filenames = []
gallery_images = []
gallery_filenames = []

for filename in os.listdir(query_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        img_path = os.path.join(query_folder, filename)
        query_filenames.append(filename)
        img = Image.open(img_path)
        query_images.append(img)

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        img_path = os.path.join(gallery_folder, filename)
        gallery_filenames.append(filename)
        img = Image.open(img_path)
        gallery_images.append(img)

print(f"Number of images in query folder: {len(query_images)}")
print(f"Number of images in gallery folder: {len(gallery_images)}")


##########
weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
model = convnext_base(weights=weights)
model.classifier[2] = torch.nn.Identity()
model = model.to(device)
preprocess = weights.transforms()
model.eval()

print("Processing query images...")
query_features = batching(query_images, batch_size=8)
print("Processing gallery images...")
gallery_features = batching(gallery_images, batch_size=32)

print("Normalizing features...")
query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

print("Computing cosine similarity matrix...")
similarity_matrix = torch.matmul(query_features, gallery_features.T)

print("Getting top 10 matches for each query...")
top_k = 10
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

top_k_filenames = []
for i in range(top_k_indices.shape[0]):
    top_k_filenames.append([gallery_filenames[idx]
                           for idx in top_k_indices[i]])

results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = top_k_filenames[i]

with open("results-convnext-base.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)
print("Saved results to results-convnext-base.json")

# Submit the results
submit(results=results, groupname="CTK",
       url="http://videosim.disi.unitn.it:3001/retrieval/")
