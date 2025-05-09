from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct # PointStruct uncommented and moved
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

client = QdrantClient(url="http://localhost:6333")

# Ensure the collection exists, if not, create it.
# This helps avoid errors if the script is run multiple times.
try:
    # Check if collection exists by trying to get its info
    collection_info = client.get_collection(collection_name="mtg_collection")
    print("Collection 'mtg_collection' already exists.")
    # Optional: Check if the existing collection's vector config matches
    existing_vector_size = collection_info.config.params.vectors.size
    desired_vector_size = 512 # For 'clip-ViT-B-32'
    if existing_vector_size != desired_vector_size:
        print(f"WARNUNG: Bestehende Sammlung hat Vektorgröße {existing_vector_size}, erwartet wird {desired_vector_size}.")
        print("Für optimale Ergebnisse sollte die Sammlung möglicherweise neu erstellt werden mit der korrekten Größe.")
        # For this script, we'll proceed but note that results might be unexpected if sizes mismatch.
except Exception as e: # Broad exception to catch cases where collection doesn't exist or other errors
    print(f"Collection 'mtg_collection' nicht gefunden oder Fehler beim Zugriff ({e}), erstelle sie.")
    client.create_collection(
        collection_name="mtg_collection",
        vectors_config=VectorParams(size=512, distance=Distance.DOT), # Adjusted for CLIP ViT-B/32
        )

    # Load the pre-trained Sentence Transformer model
    # Models like 'clip-ViT-B-32' are good for image embeddings.
    # The first time you run this, it might download the model, which can take a few minutes.
    print("Lade SentenceTransformer-Modell 'clip-ViT-B-32' (kann beim ersten Mal dauern)...")
    model = SentenceTransformer('clip-ViT-B-32')
    print("Modell geladen.")

    def image_to_vector(image_path: str, model_instance: SentenceTransformer) -> list[float] | None:
        """
        Wandelt ein Bild unter dem gegebenen Pfad in einen Vektor um,
        unter Verwendung des bereitgestellten SentenceTransformer-Modells.

        Args:
            image_path: Der Pfad zum Bild.
            model_instance: Die geladene Instanz des SentenceTransformer-Modells.

        Returns:
            Eine Liste von Floats, die den Bildvektor darstellen, oder None bei einem Fehler.
        """
        try:
            # Öffne das Bild mit Pillow
            img = Image.open(image_path)
            # Stelle sicher, dass das Bild im RGB-Format ist, falls das Modell dies erfordert
            # (CLIP-Modelle über sentence-transformers handhaben dies typischerweise gut)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Die .encode() Methode von sentence-transformers kann direkt PIL Image Objekte verarbeiten.
            vector = model_instance.encode(img).tolist()
            return vector
        except FileNotFoundError:
            print(f"Fehler: Bild nicht gefunden unter {image_path}")
            return None
        except Exception as e:
            print(f"Fehler bei der Vektorisierung von {image_path}: {e}")
            return None

    # Create Vector from img

# --- Configuration for images ---
# TODO: Erstelle einen Ordner 'images' im Verzeichnis des Skripts ('mtg-image-search/images' wenn das Skript in 'mtg-image-search/' liegt)
# und lege dort einige .jpg oder .png Bilder ab.
# Beispiel: mtg-image-search/images/card1.jpg, mtg-image-search/images/card2.png
IMAGE_SUBDIR = "images" # Unterverzeichnisname für Bilder

# Bestimme den Pfad des Skripts, um den Bilderordner relativ dazu zu finden
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # __file__ ist nicht definiert, z.B. in interaktiven Umgebungen
    script_dir = os.getcwd() # Fallback auf aktuelles Arbeitsverzeichnis

image_full_dir = os.path.join(script_dir, IMAGE_SUBDIR)

if not os.path.exists(image_full_dir):
    os.makedirs(image_full_dir)
    print(f"INFO: Verzeichnis {image_full_dir} erstellt. Bitte füge dort Bilder hinzu.")

# Hole alle Bilddateien aus dem Verzeichnis
try:
    image_files = [os.path.join(image_full_dir, f) for f in os.listdir(image_full_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
except FileNotFoundError:
    print(f"FEHLER: Das Bildverzeichnis {image_full_dir} wurde nicht gefunden, obwohl versucht wurde, es zu erstellen.")
    image_files = []


points_to_upsert = []
if not image_files:
    print(f"WARNUNG: Keine Bilder im Verzeichnis {image_full_dir} gefunden. Es werden keine Punkte hochgeladen.")
    print(f"Bitte füge Bilder (z.B. .jpg, .png) in das Verzeichnis '{image_full_dir}' und führe das Skript erneut aus.")
else:
    print(f"\nGefundene Bilder zum Verarbeiten in {image_full_dir}:")
    for path in image_files:
        print(f"- {os.path.basename(path)}")

    for idx, img_path in enumerate(image_files):
        print(f"\nVerarbeite Bild: {img_path}...")
        vector = image_to_vector(img_path, model)
        if vector:
            # Verwende den Dateinamen als Teil des Payloads für bessere Nachvollziehbarkeit
            file_name = os.path.basename(img_path)
            points_to_upsert.append(
                PointStruct(id=idx + 1, vector=vector, payload={"image_path": img_path, "file_name": file_name})
            )
            print(f"Vektor für {file_name} erstellt und für Upsert vorbereitet.")
        else:
            print(f"Konnte Vektor für {img_path} nicht erstellen. Wird übersprungen.")

# Upsert der Punkte in die Qdrant-Sammlung
if points_to_upsert:
    print(f"\nLade {len(points_to_upsert)} Vektor(en) in die Sammlung 'mtg_collection' hoch...")
    operation_info = client.upsert(
        collection_name="mtg_collection",
        wait=True, # Auf True setzen, um auf den Abschluss der Operation zu warten
        points=points_to_upsert,
    )
    print("\nUpsert-Operation abgeschlossen:")
    print(operation_info)
else:
    print("\nKeine gültigen Punkte zum Hochladen vorhanden.")

# --- Beispiel-Suche nach ähnlichen Bildern ---
if points_to_upsert:
    # Für die Suche verwenden wir den Vektor des ersten erfolgreich verarbeiteten Bildes
    # Du könntest hier auch ein separates Query-Bild verarbeiten
    query_image_path_for_search = points_to_upsert[0].payload["image_path"]
    print(f"\nSuche nach Bildern, die ähnlich sind zu: {query_image_path_for_search} (ID: {points_to_upsert[0].id})")

    query_vector_for_search = points_to_upsert[0].vector

    # Alternativ: Vektor für ein spezifisches Query-Bild generieren (auskommentiert)
    # query_image_to_search_path = "pfad/zu/deinem/suchbild.jpg" # TODO: Anpassen, falls benötigt
    # if os.path.exists(query_image_to_search_path):
    #    query_vector_for_search = image_to_vector(query_image_to_search_path,05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#         PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#         PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#         PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
#         PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#         PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#     ],
# )

# print(operation_info)

search_result = client.query_points(
    collection_name="mtg_collection",
    query=[0.2, 0.1, 0.9, 0.7],
    with_payload=False,
    limit=3
).points

print(search_result)
