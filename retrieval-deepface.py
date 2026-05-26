import os
import json
from PIL import Image
import requests
import numpy as np
from deepface import DeepFace


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


def pil_to_bgr_numpy(img):
    """
    DeepFace suele trabajar internamente con formato BGR, como OpenCV.
    PIL carga imágenes en RGB, así que convertimos RGB -> BGR.
    """
    img_rgb = np.array(img.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1]
    return img_bgr


def select_best_face_embedding(embedding_objs):
    """
    DeepFace puede devolver varias caras si detecta más de una.
    Elegimos la cara con mayor área.
    """
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    best_obj = None
    best_area = -1

    for obj in embedding_objs:
        area_info = obj.get("facial_area", {})

        w = area_info.get("w", 0)
        h = area_info.get("h", 0)
        area = w * h

        if area > best_area:
            best_area = area
            best_obj = obj

    return best_obj["embedding"]


def extract_face_embedding(img):
    """
    Extrae un embedding facial usando DeepFace con ArcFace.

    Primero intenta detectar la cara normalmente.
    Si falla, usa enforce_detection=False como fallback.
    """

    img_bgr = pil_to_bgr_numpy(img)

    try:
        embedding_objs = DeepFace.represent(
            img_path=img_bgr,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )
    except Exception:
        try:
            embedding_objs = DeepFace.represent(
                img_path=img_bgr,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True
            )
        except Exception:
            return None

    if len(embedding_objs) == 0:
        return None

    embedding = select_best_face_embedding(embedding_objs)
    embedding = np.array(embedding, dtype=np.float32)

    # L2 normalization
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    return embedding


def extract_embeddings(images, filenames):
    embeddings = []
    valid_filenames = []
    failed_filenames = []

    for i, (img, filename) in enumerate(zip(images, filenames)):
        emb = extract_face_embedding(img)

        if emb is None:
            failed_filenames.append(filename)
        else:
            embeddings.append(emb)
            valid_filenames.append(filename)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(images)} images")

    if len(embeddings) == 0:
        raise ValueError(
            "No embeddings were extracted. Check image paths or DeepFace installation.")

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)

    return embeddings, valid_filenames, failed_filenames


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
# DeepFace + ArcFace
# La primera vez puede tardar porque descarga/carga los pesos del modelo.

print("Processing query images...")
query_features, query_valid_filenames, query_failed = extract_embeddings(
    query_images,
    query_filenames
)

print("Processing gallery images...")
gallery_features, gallery_valid_filenames, gallery_failed = extract_embeddings(
    gallery_images,
    gallery_filenames
)

print(f"Valid query images: {len(query_valid_filenames)}")
print(f"Valid gallery images: {len(gallery_valid_filenames)}")
print(f"Failed query detections: {len(query_failed)}")
print(f"Failed gallery detections: {len(gallery_failed)}")

if len(query_failed) > 0:
    print("Some query images had no detected face:")
    print(query_failed[:20])

if len(gallery_failed) > 0:
    print("Some gallery images had no detected face:")
    print(gallery_failed[:20])


print("Computing cosine similarity matrix...")
# Como los embeddings ya están normalizados, dot product = cosine similarity
similarity_matrix = np.matmul(query_features, gallery_features.T)

print("Getting top 10 matches for each query...")
top_k = 10
top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]

top_k_filenames = []
for i in range(top_k_indices.shape[0]):
    top_k_filenames.append([
        gallery_valid_filenames[idx] for idx in top_k_indices[i]
    ])

results = {}

for i, query_filename in enumerate(query_valid_filenames):
    results[query_filename] = top_k_filenames[i]


# Si alguna query falla completamente, hay que devolver igualmente 10 imágenes.
# Usamos como fallback las primeras 10 imágenes válidas de gallery.
fallback_gallery = gallery_valid_filenames[:top_k]

for failed_query in query_failed:
    results[failed_query] = fallback_gallery


with open("results-deepface-arcface.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("Saved results to results-deepface-arcface.json")


# Submit the results
submit(results=results, groupname="CTK",
       url="http://videosim.disi.unitn.it:3001/retrieval/")
