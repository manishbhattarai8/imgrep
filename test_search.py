import os
import re
import torch
import json
import easyocr
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from models.image_encoder import ResNetImageEncoder
from models.text_encoder import TransformerTextEncoder
from models.dual_encoder import ImageTextRetrievalModel
from data.tokenizer import CaptionTokenizer
from rapidfuzz import fuzz
import speech_recognition as sr

# ========== CONFIG ========== #
image_folder = "data/test2014"
caption_file = "captions_train2014.json"
model_checkpoint = "checkpoints/best_model.pt"
max_len = 20
embed_dim = 256
alpha = 0.7  # Weight for semantic similarity (vs. OCR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

print(f"Using device: {device}")

# ========== LOAD TOKENIZER ========== #
print("Loading tokenizer...")
with open(caption_file, 'r') as f:
    all_captions = [a['caption'] for a in json.load(f)['annotations']]
tokenizer = CaptionTokenizer(all_captions)
print(f"Vocabulary size: {tokenizer.vocab_size()}")

# ========== LOAD MODEL ========== #
print("Initializing and loading model...")
image_encoder = ResNetImageEncoder(output_dim=embed_dim)
text_encoder = TransformerTextEncoder(vocab_size=tokenizer.vocab_size(), embed_dim=embed_dim, max_len=max_len)
model = ImageTextRetrievalModel(embed_dim=embed_dim, image_encoder=image_encoder, text_encoder=text_encoder)

try:
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded final model")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.to(device)
model.eval()

# ========== TRANSFORM ========== #
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ========== ENCODING IMAGES ========== #
def encode_images(image_folder):
    cache_path = os.path.join(cache_dir, "image_feats.pt")
    all_paths = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    encoded = []
    encoded_dict = {}

    if os.path.exists(cache_path):
        print("Loading cached image features...")
        encoded = torch.load(cache_path)
        encoded_dict = {path: feat for path, feat in encoded}

    # Remove entries for deleted images
    current_set = set(all_paths)
    cached_set = set(encoded_dict.keys())
    removed = cached_set - current_set
    for r in removed:
        del encoded_dict[r]
    if removed:
        print(f"Removed {len(removed)} deleted image(s) from cache.")

    # Encode new images
    new_paths = [p for p in all_paths if p not in encoded_dict]
    print(f"Found {len(new_paths)} new image(s) to encode.")

    with torch.no_grad():
        for path in tqdm(new_paths, desc="Encoding new images"):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                img_feat, _ = model(img_tensor, torch.zeros(1, max_len, dtype=torch.long, device=device))
                img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
                encoded_dict[path] = img_feat.squeeze(0).cpu()
            except Exception as e:
                print(f"Error with {path}: {e}")

    # Save updated cache
    final_list = [(path, feat) for path, feat in encoded_dict.items()]
    torch.save(final_list, cache_path)
    return final_list

# ========== OCR EXTRACTION ========== #
def extract_ocr_text(image_paths):
    cache_path = os.path.join(cache_dir, "ocr_index.json")
    ocr_index = {}

    if os.path.exists(cache_path):
        print("Loading cached OCR text...")
        with open(cache_path, 'r') as f:
            ocr_index = json.load(f)

    current_paths_set = set(image_paths)
    cached_paths_set = set(ocr_index.keys())

    # Remove deleted image paths
    removed = cached_paths_set - current_paths_set
    for r in removed:
        del ocr_index[r]
    if removed:
        print(f"Removed {len(removed)} deleted OCR entry(s).")

    # OCR new images
    new_paths = [p for p in image_paths if p not in ocr_index]
    print(f"Found {len(new_paths)} new image(s) for OCR.")

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    for path in tqdm(new_paths, desc="OCR"):
        try:
            text = ' '.join(reader.readtext(path, detail=0)).lower()
            ocr_index[path] = text
        except Exception as e:
            ocr_index[path] = ""

    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(ocr_index, f)

    return ocr_index

# ========== TEXT ENCODING ========== #
def encode_query(query):
    tokens = tokenizer.encode(query, max_len)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, text_feat = model(torch.zeros(1, 3, 128, 128, device=device), tokens)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    return text_feat.squeeze(0).cpu()

# ========== OCR SCORE FUNCTION ========== #

def compute_ocr_score(query, ocr_text):
    query_words = set(re.findall(r'\w+', query.lower()))
    ocr_words = set(re.findall(r'\w+', ocr_text.lower()))

    if not ocr_words:
        return 0.0

    score = fuzz.partial_ratio(query.lower(), ocr_text.lower()) / 100.0
    return score


# ========== HYBRID SEARCH ========== #
def hybrid_search(query, image_feats, ocr_index, alpha=0.5, top_k=5):
    query_feat = encode_query(query)
    scores = []
    for path, img_feat in image_feats:
        semantic_score = torch.dot(query_feat, img_feat).item()
        ocr_score = compute_ocr_score(query, ocr_index.get(path, ""))
        ocr_score *= 1.8  # amplify OCR influence
        final_score = alpha * semantic_score + (1 - alpha) * ocr_score
        scores.append((path, final_score))
        print(f"[DEBUG] {os.path.basename(path)} - semantic: {semantic_score:.4f}, ocr: {ocr_score:.4f}, final: {final_score:.4f}")

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ========== DISPLAY RESULTS ========== #
def display_results(results, show_images=True):
    if not results:
        print("No results found!")
        return
    print("\nTop results:")
    print("-" * 50)
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. {os.path.basename(path)} (score: {score:.4f})")
        if show_images:
            try:
                Image.open(path).show()
            except Exception as e:
                print(f"   Couldn't open image: {e}")

# ========== VOICE QUERY ========== #
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening... (Speak now)")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        query = recognizer.recognize_google(audio)
        print(f"🗣 You said: {query}")
        return query
    except Exception as e:
        print(f"Speech error: {e}")
        return ""

# ========== MAIN EXECUTION ========== #
if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID IMAGE-TEXT RETRIEVAL (SEMANTIC + OCR)")
    print("=" * 60)

    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' does not exist.")
        exit(1)

    image_feats = encode_images(image_folder)
    image_paths = [p for p, _ in image_feats]
    if not image_feats:
        print("❌ No images encoded. Exiting.")
        exit(1)

    ocr_index = extract_ocr_text(image_paths)

    print("\nREADY FOR SEARCH! Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            method = input("Search by (T)ext or (V)oice? ").strip().lower()
            if method == 'quit':
                break
            query = recognize_speech() if method == 'v' else input("Enter your query: ").strip()
            if not query:
                print("Empty query! Try again.")
                continue

            results = hybrid_search(query, image_feats, ocr_index, alpha=alpha, top_k=5)
            display_results(results)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
