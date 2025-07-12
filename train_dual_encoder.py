import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.image_encoder import ResNetImageEncoder
from models.text_encoder import TransformerTextEncoder
from models.dual_encoder import ImageTextRetrievalModel
from utils.losses import contrastive_loss
from data.coco_contrastive_dataset import CocoContrastiveDataset
from data.tokenizer import CaptionTokenizer
import os
import json
import time
from tqdm import tqdm
import torch.nn as nn

# Performance optimizations
os.environ['TORCH_COMPILE'] = '0'  # Disable compilation to avoid Triton issues
torch.set_float32_matmul_precision('high')  # Better performance
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
torch.backends.cudnn.allow_tf32 = True

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir="checkpoints"):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Save numbered checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        numbered_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, numbered_path)
    
    return latest_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if it exists
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('loss', float('inf'))
    
    print(f"✓ Resumed from epoch {start_epoch + 1}, loss: {best_loss:.4f}")
    return start_epoch, best_loss

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent checkpoint"""
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(latest_path):
        return latest_path
    
    # Look for numbered checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            return os.path.join(checkpoint_dir, checkpoints[-1])
    
    return None

def main():
    # Configuration - OPTIMIZED FOR SPEED
    image_folder = "data/train2014"
    caption_file = "captions_train2014.json"
    batch_size = 64  # Increased from 32 - try 128 if you have enough GPU memory
    embed_dim = 256
    epochs = 10
    lr = 2e-4  # Increased LR since we increased batch size
    max_len = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoints"
    
    # Resume training flag
    resume_training = True
    
    # Enable mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    print(f"Using device: {device}")
    print(f"Mixed precision training: {use_amp}")
    print(f"Resume training: {resume_training}")
    print(f"Batch size: {batch_size}")
    print(f"Total epochs: {epochs}")
    print("=" * 50)

    # Prepare tokenizer
    print("Loading captions and preparing tokenizer...")
    with open(caption_file, 'r') as f:
        all_captions = [a['caption'] for a in json.load(f)['annotations']]
    tokenizer = CaptionTokenizer(all_captions)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")

    # Define transforms - OPTIMIZED
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader - OPTIMIZED FOR SPEED
    print("Loading dataset...")
    dataset = CocoContrastiveDataset(image_folder, caption_file, tokenizer, transform, max_len)
    
    # Calculate optimal num_workers (usually 4 * num_gpus, but max 8-12)
    num_workers = min(8, os.cpu_count())
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,  # Increased workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # Prefetch more batches
        drop_last=True  # Ensures consistent batch sizes
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Using {num_workers} workers for data loading")

    # Models
    print("Initializing models...")
    image_encoder = ResNetImageEncoder(output_dim=256)
    text_encoder = TransformerTextEncoder(vocab_size=tokenizer.vocab_size(), embed_dim=256, max_len=max_len)
    model = ImageTextRetrievalModel(embed_dim=embed_dim, image_encoder=image_encoder, text_encoder=text_encoder)
    model = model.to(device)
    
    # Enable model optimizations
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
    
    # Safe model compilation
    try:
        if hasattr(torch, 'compile') and os.environ.get('TORCH_COMPILE') != '0':
            model = torch.compile(model)
            print("✓ Model compiled for faster training")
        else:
            print("⚠ Model compilation disabled")
    except Exception as e:
        print(f"⚠ Model compilation failed: {e}")
        print("Continuing without compilation...")

    # Optimizer and scheduler - OPTIMIZED
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01,
        eps=1e-6,  # Slightly smaller eps for stability
        fused=True if hasattr(torch.optim.AdamW, 'fused') else False  # Fused optimizer if available
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_training:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path:
            start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        else:
            print("No checkpoint found, starting fresh training")
    else:
        print("Starting fresh training (resume_training=False)")
    
    # Test model dimensions
    print("Testing model dimensions...")
    try:
        dummy_images = torch.randn(2, 3, 224, 224).to(device)
        dummy_captions = torch.randint(0, tokenizer.vocab_size(), (2, max_len)).to(device)
        with torch.no_grad():
            image_feats, text_feats = model(dummy_images, dummy_captions)
            print(f"✓ Model test passed! Image features: {image_feats.shape}, Text features: {text_feats.shape}")
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return

    # Gradient clipping and accumulation
    max_grad_norm = 1.0
    accumulation_steps = 1  # Increase if you want to simulate larger batch sizes
    
    if start_epoch < epochs:
        print(f"Starting training from epoch {start_epoch + 1}/{epochs}...")
        print("=" * 50)
    else:
        print("Training already completed!")
        return

    # Training loop - OPTIMIZED
    training_start_time = time.time()
    
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0
            epoch_start_time = time.time()
            
            # Progress bar with better formatting
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', 
                               leave=False, ncols=100, 
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for batch_idx, (images, captions) in enumerate(progress_bar):
                # Move to device with non_blocking
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        image_feats, text_feats = model(images, captions)
                        loss = contrastive_loss(image_feats, text_feats) / accumulation_steps
                    
                    # Backward pass with scaling
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard forward pass
                    image_feats, text_feats = model(images, captions)
                    loss = contrastive_loss(image_feats, text_feats) / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item() * accumulation_steps
                batch_count += 1
                
                # Update progress bar less frequently for speed
                if batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item() * accumulation_steps:.3f}',
                        'Avg': f'{total_loss/batch_count:.3f}',
                        'LR': f'{current_lr:.1e}'
                    })

            # Step scheduler
            scheduler.step()
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*20} EPOCH {epoch+1} SUMMARY {'='*20}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.1f} min)")
            print(f"Samples per second: {len(dataset)/(epoch_time):.1f}")
            print(f"Batches per second: {len(dataloader)/(epoch_time):.1f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_dir)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, best_model_path)
                print(f"✓ New best model saved! Loss: {avg_loss:.4f}")
            
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Training interrupted by user!")
        print("Saving checkpoint before exit...")
        checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_dir)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print("You can resume training later by running the script again.")
        print("="*50)
        return

    # Training completed
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 Training completed!")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    print(f"Best loss achieved: {best_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    main()