# TensorFlow GPU development environment

A ready-to-use TensorFlow environment with NVIDIA GPU support for VS Code. Designed for cross-platform support and wide GPU compatibility.

## What's included

| Category | Versions |
|----------|----------|
| **GPU** | CUDA 12.5, cuDNN 9.1 |
| **ML** | TensorFlow 2.16, Keras 3.3, Scikit-learn 1.4 |
| **Python** | Python 3.10, NumPy 1.24, Pandas 2.2, Matplotlib 3.10 |
| **Tools** | JupyterLab, TensorBoard |

Based on [NVIDIA's TensorFlow 24.06 container](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-06.html).

> **No NVIDIA GPU?** Use the CPU version instead: [gperdrizet/tensorflow-CPU](https://github.com/gperdrizet/tensorflow-CPU)

## Project structure

```
tensorflow-GPU/
├── .devcontainer/
│   └── devcontainer.json       # Dev container configuration
├── data/                       # Store datasets here
├── logs/                       # TensorBoard logs
├── models/                     # Saved model files
├── notebooks/
│   ├── environment_test.ipynb  # Verify your setup
│   └── functions/              # Helper modules for notebooks
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- **NVIDIA GPU** (Pascal or newer) with driver ≥545
- **Docker** with GPU support ([Windows](https://docs.docker.com/desktop/setup/install/windows-install) | [Linux](https://docs.docker.com/desktop/setup/install/linux))
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

> **Linux users:** Also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### GPU compatibility

This environment requires an NVIDIA GPU with **compute capability 6.0+** (Pascal architecture or newer):

| Architecture | Example GPUs | Compute Capability |
|--------------|--------------|-------------------|
| Pascal | GTX 1050–1080, Tesla P100 | 6.0–6.1 |
| Volta | Tesla V100, Titan V | 7.0 |
| Turing | RTX 2060–2080, GTX 1660 | 7.5 |
| Ampere | RTX 3060–3090, A100 | 8.0–8.6 |
| Ada Lovelace | RTX 4060–4090 | 8.9 |
| Hopper | H100, H200 | 9.0 |
| Blackwell | RTX 5070–5090, B100, B200 | 10.0 |

Check your GPU's compute capability: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

> **Note:** This environment is configured for broad GPU compatibility, supporting Pascal and newer architectures. If you have a recent GPU (Ada Lovelace, Hopper, or Blackwell), you may benefit from using a newer CUDA version to access the latest performance optimizations and features. Consider setting up a custom environment with an updated [NVIDIA TensorFlow container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) to take full advantage of your hardware.

## Quick start

1. **Fork** this repository (click "Fork" button above)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/tensorflow-GPU.git
   ```

3. **Open VS Code**

4. **Open Folder in Container** from the VS Code command pallet (Ctrl+shift+p), start typing `Open Folder in`...

5. **Verify** by running `notebooks/environment_test.ipynb`

## Using as a template for new projects

You can use your fork as a starting point for new TensorFlow projects by setting it up as a GitHub template repository:

1. **Go to your fork** on GitHub

2. **Open Settings** → **General**

3. **Check** "Template repository" under the repository name

4. **Create new projects** by clicking "Use this template" → "Create a new repository" from your fork's main page

This creates a fresh repository with all the dev container configuration, without copying the git history.

## Adding Python packages

### Using pip directly

Install packages in the container terminal:

```bash
pip install <package-name>
```

> **Note:** Packages installed this way will be lost when the container is rebuilt.

### Using requirements.txt (Recommended)

For persistent packages that survive container rebuilds:

1. **Create** a `requirements.txt` file in the repository root:
   ```
   scikit-image==0.22.0
   plotly
   ```

2. **Update** `.devcontainer/devcontainer.json` to install packages on container creation by adding a `postCreateCommand`:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. **Rebuild** the container (`F1` → "Dev Containers: Rebuild Container")

Now your packages will be automatically installed whenever the container is created.

## TensorBoard

To launch TensorBoard:

1. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Python: Launch TensorBoard**
3. Select the `logs/` directory when prompted

TensorBoard will open in a new tab within VS Code. Place your training logs in the `logs/` directory.

## Keeping your fork updated

```bash
# Add upstream (once)
git remote add upstream https://github.com/gperdrizet/tensorflow-GPU.git

# Sync
git fetch upstream
git merge upstream/main
```

