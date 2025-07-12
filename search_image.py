import torch
from models.image_encoder import ResNetImageEncoder
from models.text_encoder import TransformerTextEncoder
from models.dual_encoder import ImageTextRetrievalModel
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
from data.tokenizer import CaptionTokenizer
import speech_recognition as sr

# Configs - UPDATED TO MATCH TRAINING
image_folder = "data/test2014"
caption_file = "captions_train2014.json"  # still used to build tokenizer
model_checkpoint = "checkpoints/best_model.pt"  # Updated path to match training
max_len = 20  # Changed from 40 to 20 to match training
embed_dim = 256  # Matches training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Loading model from: {model_checkpoint}")

# Load tokenizer
print("Loading tokenizer...")
with open(caption_file, 'r') as f:
    all_captions = [a['caption'] for a in json.load(f)['annotations']]
tokenizer = CaptionTokenizer(all_captions)
print(f"Vocabulary size: {tokenizer.vocab_size()}")

# Load model - CORRECTED DIMENSIONS
print("Initializing model...")
image_encoder = ResNetImageEncoder(output_dim=256)  # Changed from 512 to 256
text_encoder = TransformerTextEncoder(vocab_size=tokenizer.vocab_size(), embed_dim=256, max_len=max_len)  # Changed from 512 to 256
model = ImageTextRetrievalModel(embed_dim=embed_dim, image_encoder=image_encoder, text_encoder=text_encoder)

# Load checkpoint properly (handle different checkpoint formats)
print("Loading trained model...")
try:
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        # Loading from training checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Model loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        # Loading from final model save
        model.load_state_dict(checkpoint)
        print("✓ Loaded final model")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the checkpoint path is correct and the model was trained successfully.")
    exit(1)

model.to(device)
model.eval()

# Image transform - UPDATED TO MATCH TRAINING
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Changed from (224, 224) to (128, 128) to match training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Encode test images
def encode_images(image_folder):
    """Encode all images in the folder"""
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' not found!")
        return []
    
    image_paths = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return []
    
    print(f"Found {len(image_paths)} images to encode...")
    
    all_feats = []
    model.eval()
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Encoding images"):
            try:
                image = Image.open(path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                
                # Get image features using the model's forward pass
                image_feat, _ = model(image, torch.zeros(1, max_len, dtype=torch.long, device=device))
                image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
                
                all_feats.append((path, image_feat.squeeze(0).cpu()))
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
    
    print(f"Successfully encoded {len(all_feats)} images")
    return all_feats

# Encode user query
def encode_query(query):
    """Encode text query into feature vector"""
    tokens = tokenizer.encode(query, max_len)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get text features using the model's forward pass
        _, text_feat = model(torch.zeros(1, 3, 128, 128, device=device), tokens)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    
    return text_feat.squeeze(0).cpu()

# Retrieve top matches
def search(query, image_feats, top_k=5):
    """Search for images matching the query"""
    if not image_feats:
        print("No image features available!")
        return []
    
    print(f"Searching for: '{query}'")
    query_feat = encode_query(query)
    
    # Calculate similarities
    sims = []
    for path, feat in image_feats:
        similarity = torch.dot(query_feat, feat).item()
        sims.append((path, similarity))
    
    # Sort by similarity (highest first)
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# Display results
def display_results(results, show_images=True):
    """Display search results"""
    if not results:
        print("No results found!")
        return
    
    print(f"\nTop {len(results)} results:")
    print("-" * 50)
    
    for i, (path, score) in enumerate(results, 1):
        filename = os.path.basename(path)
        print(f"{i}. {filename} (similarity: {score:.4f})")
        
        if show_images:
            try:
                Image.open(path).show()
            except Exception as e:
                print(f"   Error displaying image: {e}")
                
def recognize_speech():
    """Capture and return spoken query using microphone"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening... (Speak now)")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    
    try:
        query = recognizer.recognize_google(audio)
        print(f"🗣 You said: {query}")
        return query
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results; {e}")
    except Exception as e:
        print(f"⚠️ Error with microphone input: {e}")
    return ""


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGE-TEXT RETRIEVAL SYSTEM")
    print("=" * 60)
    
    # Check if test images exist
    if not os.path.exists(image_folder):
        print(f"Error: Test image folder '{image_folder}' not found!")
        print("Please make sure the path is correct.")
        exit(1)
    
    # Encode all test images
    print("Encoding test images...")
    image_feats = encode_images(image_folder)
    
    if not image_feats:
        print("No images were successfully encoded. Exiting.")
        exit(1)
    
    # Interactive search loop
    print("\n" + "=" * 60)
    print("READY FOR SEARCH!")
    print("=" * 60)
    print("Enter your search queries below.")
    print("Type 'quit' to exit, 'help' for examples.")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\nChoose input method - (T)ext or (V)oice? ").strip().lower()

            if choice == 'quit':
                print("Goodbye!")
                break
            elif choice == 'help':
                print("\nExample queries:")
                print("- 'a cat sitting on a table'")
                print("- 'person riding a bicycle'")
                print("- 'red car on the street'")
                print("- 'dog playing in the park'")
                continue
            elif choice == 'v':
                query = recognize_speech()
            elif choice == 't':
                query = input("Type your search query: ").strip()
            else:
                print("Invalid input method. Type 'T' for text or 'V' for voice.")
                continue

            if not query:
                print("❌ No query entered. Try again.")
                continue

            print(f"Searching for: '{query}'")
            results = search(query, image_feats, top_k=5)
            display_results(results, show_images=True)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            continue
            