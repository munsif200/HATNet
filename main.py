"""
ReST: Reconfigurable Spatial-Temporal Graph Model
Main entry point for training and testing
"""

import argparse
import os
import sys
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs import cfg
from src.trainer import ReSTTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ReST: Reconfigurable Spatial-Temporal Graph Model")
    parser.add_argument(
        "--config_file", 
        default="", 
        help="path to config file", 
        type=str
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "demo"],
        help="Mode to run the model in"
    )
    parser.add_argument(
        "--solver_type",
        default="SG",
        choices=["SG", "TG"],
        help="Solver type: SG (Spatial Graph) or TG (Temporal Graph)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None,
        nargs=argparse.REMAINDER
    )
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration"""
    # Load config file if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            cfg.merge_from_file(args.config_file)
        else:
            logger.warning(f"Config file not found: {args.config_file}")
    
    # Override with command line arguments
    if args.mode:
        cfg.MODEL.MODE = args.mode
    
    if args.solver_type:
        cfg.SOLVER.TYPE = args.solver_type
    
    if args.device:
        cfg.MODEL.DEVICE = args.device
    
    if args.epochs:
        cfg.SOLVER.EPOCHS = args.epochs
    
    if args.batch_size:
        cfg.SOLVER.BATCH_SIZE = args.batch_size
    
    if args.lr:
        cfg.SOLVER.LR = args.lr
    
    # Merge from command line opts
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Freeze config
    cfg.freeze()
    
    # Set environment variable for CUDA
    if cfg.MODEL.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    return cfg


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Log configuration
    logger.info("Starting ReST Model")
    logger.info(f"Mode: {config.MODEL.MODE}")
    logger.info(f"Solver Type: {config.SOLVER.TYPE}")
    logger.info(f"Device: {config.MODEL.DEVICE}")
    
    # Initialize trainer
    trainer = ReSTTrainer(config)
    
    # Run based on mode
    if config.MODEL.MODE == 'train':
        logger.info("Starting training...")
        trainer.train()
    elif config.MODEL.MODE == 'test':
        logger.info("Starting testing...")
        results = trainer.test()
        logger.info(f"Test Results: {results}")
    elif config.MODEL.MODE == 'demo':
        logger.info("Running demonstration...")
        trainer.demonstrate()
    else:
        logger.error(f"Unknown mode: {config.MODEL.MODE}")
        sys.exit(1)
    
    logger.info("ReST Model execution completed!")


if __name__ == "__main__":
    main()