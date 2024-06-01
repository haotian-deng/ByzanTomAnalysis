# ByzanTomAnalysis

## Overview
ByzanTomAnalysis is a comprehensive analysis platform specifically developed for Network Boolean Tomography (NBT) and Byzantine attack research. The main purpose of this platform is to help researchers and network administrators better understand and analyze network states and their changes by providing an intuitive graphical interface that showcases network topology, the application effects of NBT technology, and the impact of Byzantine attacks.

## Features
- Visualize network topologies using various input formats.
- Perform different topology-related algorithms and analyses.
- Export and save results in different formats.

## Requirements
- Python 3.10 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the project directory:**
    ```bash
    cd ByzanTomAnalysis
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Extract the dataset (if necessary):**
    
    ```bash
    unzip dataset.zip -d topology
    ```

## Usage
Run the main script to start the application:

```bash
python main.py
```

## Directory Structure

```
ByzanTomAnalysis/
│
├── algorithm/                     # Directory containing topology algorithms
│   ├── alg_scfs.py
│   ├── alg_map.py
│   └── alg_clink.py
│
├── image/                         # Directory containing image assets
│   └── demo.png
│
├── topology/                      # Directory containing sample topology files
│   ├── demoGML.gml
│   ├── demoAdjacencyMatrix.csv
│   └── demoRoutineMatrix.csv
│
├── cache/  										   # Used to store cache generated during running
├── dataset.zip                    # Compressed dataset file
├── main.py                        # Main script to run the application
├── Topo_GUI.py                    # GUI implementation script
├── requirements.txt               # File listing required Python packages
└── README.md                      # This README file
```

## Notes

- Ensure that the dataset is properly extracted before running the application.
- For any issues or bugs, please report to the repository's issue tracker.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the contributors and the PySimpleGUI community for their support and resources.

