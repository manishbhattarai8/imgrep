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

# Configs - UPDATED TO MATCH TRAINING
image_folder = "data/test2014"
caption_file = "captions_train2014.json"  # still used to build tokenizer
checkpoint_dir = "checkpoints"  # Directory containing checkpoints
max_len = 20  # Changed from 40 to 20 to match training
embed_dim = 256  # Matches training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_available_checkpoints(checkpoint_dir="checkpoints"):
    """Find all available checkpoints and return them sorted by preference"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    
    # Look for specific checkpoint types
    best_model = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model):
        checkpoints.append(("best_model", best_model, "Best performing model"))
    
    final_model = os.path.join(checkpoint_dir, "final_model.pt")
    if os.path.exists(final_model):
        checkpoints.append(("final_model", final_model, "Final epoch model"))
    
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint):
        checkpoints.append(("latest_checkpoint", latest_checkpoint, "Latest checkpoint"))
    
    # Look for numbered epoch checkpoints
    epoch_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if epoch_files:
        # Sort by epoch number
        epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for epoch_file in epoch_files:
            epoch_num = int(epoch_file.split('_')[-1].split('.')[0])
            full_path = os.path.join(checkpoint_dir, epoch_file)
            checkpoints.append((f"epoch_{epoch_num}", full_path, f"Epoch {epoch_num} checkpoint"))
    
    return checkpoints

def select_checkpoint(checkpoint_dir="checkpoints"):
    """Interactive checkpoint selection"""
    available_checkpoints = find_available_checkpoints(checkpoint_dir)
    
    if not available_checkpoints:
        print(f"No checkpoints found in '{checkpoint_dir}'")
        return None
    
    print(f"\nAvailable checkpoints in '{checkpoint_dir}':")
    print("-" * 60)
    
    for i, (name, path, description) in enumerate(available_checkpoints, 1):
        # Get checkpoint info if possible
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'epoch' in checkpoint and 'loss' in checkpoint:
                epoch_info = f"Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}"
            else:
                epoch_info = "Final model"
        except:
            epoch_info = "Unknown"
        
        print(f"{i}. {description}")
        print(f"   File: {os.path.basename(path)}")
        print(f"   Info: {epoch_info}")
        print()
    
    # Auto-select best model if available, otherwise ask user
    if len(available_checkpoints) == 1:
        print(f"Using only available checkpoint: {available_checkpoints[0][2]}")
        return available_checkpoints[0][1]
    
    # Check if best_model exists and auto-select it
    best_model_exists = any(name == "best_model" for name, _, _ in available_checkpoints)
    if best_model_exists:
        print("Auto-selecting best model (use 'manual' to choose manually)")
        choice = input("Press Enter to use best model, or type 'manual' for manual selection: ").strip()
        if choice.lower() != 'manual':
            return next(path for name, path, _ in available_checkpoints if name == "best_model")
    
    # Manual selection
    while True:
        try:
            choice = input(f"Select checkpoint (1-{len(available_checkpoints)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available_checkpoints):
                selected = available_checkpoints[int(choice) - 1]
                print(f"Selected: {selected[2]}")
                return selected[1]
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None

# Encode test images
def encode_images(image_folder, model, transform):
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
def encode_query(query, model, tokenizer):
    """Encode text query into feature vector"""
    tokens = tokenizer.encode(query, max_len)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get text features using the model's forward pass
        _, text_feat = model(torch.zeros(1, 3, 128, 128, device=device), tokens)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    
    return text_feat.squeeze(0).cpu()

# Retrieve top matches
def search(query, image_feats, model, tokenizer, top_k=5):
    """Search for images matching the query"""
    if not image_feats:
        print("No image features available!")
        return []
    
    print(f"Searching for: '{query}'")
    query_feat = encode_query(query, model, tokenizer)
    
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

def main():
    print(f"Using device: {device}")
    
    # Select checkpoint
    model_checkpoint = select_checkpoint(checkpoint_dir)
    if model_checkpoint is None:
        print("No checkpoint selected. Exiting.")
        return
    
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
            if 'loss' in checkpoint:
                print(f"✓ Model loss: {checkpoint['loss']:.4f}")
            if 'timestamp' in checkpoint:
                import time
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp']))
                print(f"✓ Saved at: {timestamp}")
        else:
            # Loading from final model save
            model.load_state_dict(checkpoint)
            print("✓ Loaded final model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the checkpoint path is correct and the model was trained successfully.")
        return
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully and ready for inference!")
    
    # Image transform - UPDATED TO MATCH TRAINING
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Changed from (224, 224) to (128, 128) to match training
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    print("=" * 60)
    print("IMAGE-TEXT RETRIEVAL SYSTEM")
    print("=" * 60)
    
    # Check if test images exist
    if not os.path.exists(image_folder):
        print(f"Error: Test image folder '{image_folder}' not found!")
        print("Please make sure the path is correct.")
        return
    
    # Encode all test images
    print("\nEncoding test images...")
    image_feats = encode_images(image_folder, model, transform)
    
    if not image_feats:
        print("No images were successfully encoded. Exiting.")
        return
    
    # Interactive search loop
    print("\n" + "=" * 60)
    print("READY FOR SEARCH!")
    print("=" * 60)
    print("Enter your search queries below.")
    print("Type 'quit' to exit, 'help' for examples, 'stats' for model info.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nExample queries:")
                print("- 'a cat sitting on a table'")
                print("- 'person riding a bicycle'")
                print("- 'red car on the street'")
                print("- 'dog playing in the park'")
                print("- 'blue sky with clouds'")
                continue
            elif query.lower() == 'stats':
                print(f"\nModel Information:")
                print(f"- Checkpoint: {os.path.basename(model_checkpoint)}")
                print(f"- Vocabulary size: {tokenizer.vocab_size()}")
                print(f"- Max text length: {max_len}")
                print(f"- Embedding dimension: {embed_dim}")
                print(f"- Number of encoded images: {len(image_feats)}")
                continue
            elif not query:
                print("Please enter a search query.")
                continue
            
            # Perform search
            results = search(query, image_feats, model, tokenizer, top_k=5)
            display_results(results, show_images=True)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            continue

if __name__ == "__main__":
    main()