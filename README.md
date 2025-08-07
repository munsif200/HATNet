# HATNet: Hierarchical Attention Transformer Network

HATNet is a DL framework specifically developed to optimize the synthesis of organic and inorganic materials, including **molybdenum disulfide (MoS₂)**, and to estimate the **photoluminescent quantum yield (PLQY)**. By leveraging the power of the **multi-head attention (MHA) mechanism**, HATNet captures complex dependencies within feature spaces, offering a significant advancement over traditional models like XGBoost and Support Vector Machines (SVMs). This unified framework, designed for classification and regression tasks, achieved state-of-the-art performance in material synthesis optimization MoS₂ and PLQY.

## ReST Implementation

This repository now includes an implementation of **ReST: A Reconfigurable Spatial-Temporal Graph Model** based on the ICCV 2023 paper by [chengche6230](https://github.com/chengche6230/ReST). ReST is a novel approach for multi-camera multi-object tracking that uses reconfigurable spatial-temporal graphs.

### Key Features

- **Spatial Graph (SG)**: Associates objects across multiple spatial views/cameras
- **Temporal Graph (TG)**: Tracks objects across time using temporal associations  
- **Message Passing Networks (MPN)**: Graph neural networks for feature propagation
- **Unified Framework**: Combines spatial and temporal reasoning in a single model

### ReST Architecture

ReST implements a two-stage association approach:

1. **Spatial Association**: First associates all detected objects across cameras spatially
2. **Temporal Association**: Then reconfigures into a temporal graph for tracking across time

This approach enables robust spatial and temporal-aware feature extraction while addressing fragmented tracklet problems.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/munsif200/HATNet.git
cd HATNet

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Demo Mode
```bash
# Spatial Graph demo
python main.py --mode demo --solver_type SG --device cpu

# Temporal Graph demo  
python main.py --mode demo --solver_type TG --device cpu
```

#### Training
```bash
# Train spatial graph
python main.py --mode train --solver_type SG --epochs 50 --batch_size 1

# Train temporal graph
python main.py --mode train --solver_type TG --epochs 50 --batch_size 1
```

#### Testing
```bash
# Test trained model
python main.py --mode test --solver_type SG
```

### Configuration

The model can be configured using YAML files or command line arguments:

```yaml
# configs/demo.yml
MODEL:
  DEVICE: "cuda"
  MODE: "demo"

SOLVER:
  TYPE: "SG"  # or "TG"
  EPOCHS: 100
  BATCH_SIZE: 4
  LR: 0.001

GRAPH:
  NODE_DIM: 32
  EDGE_DIM: 6
  MESSAGE_DIM: 32
```

### Project Structure

```
HATNet/
├── configs/                 # Configuration files
│   ├── default.py          # Default configuration
│   └── demo.yml            # Demo configuration
├── src/
│   ├── models/             # ReST model implementations
│   │   ├── mpn.py          # Message Passing Network
│   │   ├── spatial_graph.py # Spatial Graph Model
│   │   ├── temporal_graph.py # Temporal Graph Model
│   │   └── rest_model.py    # Main ReST Model
│   ├── utils/              # Utilities
│   │   ├── losses.py       # Loss functions
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── data_loader.py  # Data loading utilities
│   └── trainer.py          # Training and testing pipeline
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Components

### Message Passing Network (MPN)
The core graph neural network that processes node and edge features through multiple layers of message passing with residual connections.

### Spatial Graph
Handles spatial associations between objects detected across different cameras or views. Creates fully-connected graphs to reason about spatial relationships.

### Temporal Graph  
Manages temporal associations by connecting objects across consecutive time steps. Uses LSTM layers for sequence modeling and trajectory generation.

### ReST Model
The main model that combines spatial and temporal graphs, supporting different operation modes:
- `SG`: Spatial Graph only
- `TG`: Temporal Graph (includes spatial processing)
- Joint: Combined spatial-temporal processing

## Applications

### Material Science (HATNet)
- Advanced material synthesis optimization
- Photoluminescent quantum yield estimation
- Multi-modal material property prediction

### Computer Vision (ReST)
- Multi-camera object tracking
- Spatial-temporal association learning
- Trajectory prediction and analysis

## Key Highlights

- **Unified Framework**: Combines classification and regression tasks using shared attention-based architecture
- **State-of-the-Art Performance**: Achieves 95% classification accuracy for MoS₂ synthesis and lower MSE values for PLQY estimation  
- **Automated Feature Learning**: Eliminates manual feature engineering by capturing intricate feature interactions
- **Flexible Architecture**: Supports both spatial-only and temporal tracking modes

## Code Availability

**The original HATNet code will be made publicly available following the acceptance of the associated research paper. Updates regarding the release will be provided on this repository**.

This ReST implementation is available now and can serve as a foundation for spatial-temporal modeling in various domains.

## Citation

If you use this ReST implementation, please cite:

```bibtex
@InProceedings{Cheng_2023_ICCV,
    author    = {Cheng, Cheng-Che and Qiu, Min-Xuan and Chiang, Chen-Kuo and Lai, Shang-Hong},
    title     = {ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10051-10060}
}
```

## Acknowledgment

This research was supported by the Nano & Material Technology Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT.  
**Project Name**: Data HUB for Solid Electrolyte Materials Based on SyncroLab Data Cloud  
**Grant Number**: RS-2024-00446825

Special thanks to [chengche6230](https://github.com/chengche6230) for the original ReST implementation that inspired this work.

## License

Upon publication, this project will be released under an open-source license, ensuring accessibility to the research and development community.

