"""
ReST Trainer: Training and testing pipeline for the ReST model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from src.models import ReSTModel
from src.utils.losses import ReSTLoss
from src.utils.metrics import compute_metrics
from src.utils.data_loader import SyntheticDataset


class ReSTTrainer:
    """
    Trainer class for ReST model
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        
        # Create output directories
        os.makedirs(cfg.OUTPUT.LOG_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT.CKPT_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT.VIS_DIR, exist_ok=True)
        
        # Initialize model
        self.model = ReSTModel(cfg)
        
        # Initialize loss function
        self.criterion = ReSTLoss(cfg)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        
        # Initialize scheduler
        if cfg.SOLVER.SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.SOLVER.EPOCHS
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        logger.info(f"Initialized ReST Trainer for {cfg.SOLVER.TYPE} mode")
    
    def train(self):
        """
        Training loop for ReST model
        """
        logger.info("Starting training...")
        
        # Create synthetic dataset for demonstration
        train_dataset = SyntheticDataset(
            num_samples=1000,
            mode=self.cfg.SOLVER.TYPE,
            device=self.device
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Set to 0 for synthetic data
        )
        
        self.model.train()
        
        for epoch in range(self.cfg.SOLVER.EPOCHS):
            epoch_loss = 0.0
            epoch_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.SOLVER.EPOCHS}") as pbar:
                for batch_idx, batch_data in enumerate(pbar):
                    # Move data to device
                    batch_data = self._move_to_device(batch_data)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    
                    # Compute loss
                    loss = self.criterion(outputs, batch_data['targets'])
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    metrics = compute_metrics(outputs, batch_data['targets'])
                    for key in epoch_metrics:
                        epoch_metrics[key] += metrics.get(key, 0.0)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{metrics.get('accuracy', 0.0):.3f}"
                    })
            
            # Average metrics over epoch
            epoch_loss /= len(train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_loader)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                       f"Acc={epoch_metrics['accuracy']:.3f}, "
                       f"Prec={epoch_metrics['precision']:.3f}, "
                       f"Rec={epoch_metrics['recall']:.3f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, epoch_loss)
        
        logger.info("Training completed!")
    
    def test(self):
        """
        Testing loop for ReST model
        """
        logger.info("Starting testing...")
        
        # Load checkpoint if specified
        if self.cfg.SOLVER.TYPE == 'SG' and self.cfg.TEST.CKPT_FILE_SG:
            self.load_checkpoint(self.cfg.TEST.CKPT_FILE_SG)
        elif self.cfg.SOLVER.TYPE == 'TG' and self.cfg.TEST.CKPT_FILE_TG:
            self.load_checkpoint(self.cfg.TEST.CKPT_FILE_TG)
        
        # Create test dataset
        test_dataset = SyntheticDataset(
            num_samples=200,
            mode=self.cfg.SOLVER.TYPE,
            device=self.device
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Test one by one
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        total_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
                # Move data to device
                batch_data = self._move_to_device(batch_data)
                
                # Forward pass
                outputs = self.model(batch_data)
                
                # Compute metrics
                metrics = compute_metrics(outputs, batch_data['targets'])
                for key in total_metrics:
                    total_metrics[key] += metrics.get(key, 0.0)
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= len(test_loader)
        
        logger.info(f"Test Results: Acc={total_metrics['accuracy']:.3f}, "
                   f"Prec={total_metrics['precision']:.3f}, "
                   f"Rec={total_metrics['recall']:.3f}")
        
        return total_metrics
    
    def _move_to_device(self, batch_data):
        """Move batch data to the specified device"""
        device_data = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_data[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_data[key] = self._move_to_device(value)
            else:
                device_data[key] = value
        return device_data
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.cfg
        }
        
        ckpt_path = os.path.join(
            self.cfg.OUTPUT.CKPT_DIR,
            f"rest_{self.cfg.SOLVER.TYPE}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path):
        """Load model checkpoint"""
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")
    
    def demonstrate(self):
        """
        Demonstrate the ReST model with synthetic data
        """
        logger.info("Running ReST demonstration...")
        
        # Create a simple demonstration
        demo_data = self._create_demo_data()
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(demo_data)
            
            logger.info("Demo Results:")
            logger.info(f"Output keys: {list(outputs.keys())}")
            
            if 'spatial_outputs' in outputs:
                spatial_scores = outputs['spatial_outputs']['association_scores']
                logger.info(f"Spatial associations shape: {spatial_scores.shape}")
                logger.info(f"Sample spatial scores (first 3x3):")
                logger.info(f"{spatial_scores[:3, :3]}")
                
            if 'trajectories' in outputs:
                trajectories = outputs['trajectories']
                logger.info(f"Found {len(trajectories)} trajectories")
                for i, traj in enumerate(trajectories):
                    logger.info(f"Trajectory {i}: {traj}")
            
            if 'final_embeddings' in outputs:
                final_emb = outputs['final_embeddings']
                logger.info(f"Final embeddings shape: {final_emb.shape}")
                logger.info(f"Embedding mean: {final_emb.mean().item():.4f}, std: {final_emb.std().item():.4f}")
            
            if 'node_embeddings' in outputs:
                node_emb = outputs['node_embeddings']
                logger.info(f"Node embeddings shape: {node_emb.shape}")
                logger.info(f"Node embedding mean: {node_emb.mean().item():.4f}")
            
            if 'association_scores' in outputs:
                assoc_scores = outputs['association_scores']
                logger.info(f"Association scores shape: {assoc_scores.shape}")
                logger.info(f"Sample association scores: {assoc_scores[:3, :3]}")
    
    def _create_demo_data(self):
        """Create demonstration data"""
        if self.cfg.SOLVER.TYPE == 'SG':
            # Spatial graph demo
            num_objects = 5
            spatial_features = torch.randn(num_objects, self.cfg.ST.SPATIAL_DIM, device=self.device)
            positions = torch.randn(num_objects, 2, device=self.device) * 10
            
            return {
                'spatial_features': spatial_features,
                'positions': positions,
                'targets': torch.eye(num_objects, device=self.device)
            }
        
        else:
            # Temporal graph demo
            time_steps, num_objects = 5, 3
            spatial_features = torch.randn(time_steps, num_objects, self.cfg.ST.SPATIAL_DIM, device=self.device)
            positions = torch.randn(time_steps, num_objects, 2, device=self.device) * 10
            
            # Simple temporal info
            temporal_info = {
                'velocities': torch.randn(time_steps, num_objects, 2, device=self.device),
                'distances': torch.randn(time_steps-1, num_objects, num_objects, device=self.device),
                'appearance_sim': torch.ones(time_steps-1, num_objects, num_objects, device=self.device),
                'confidences': torch.ones(time_steps, num_objects, device=self.device)
            }
            
            return {
                'spatial_features': spatial_features,
                'positions': positions,
                'temporal_info': temporal_info,
                'targets': torch.eye(num_objects, device=self.device)
            }