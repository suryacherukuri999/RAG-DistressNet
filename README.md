# RAG-DistressNet

A Retrieval-Augmented Generation (RAG) system that supports searching through **PDFs** and **Images** using vector embeddings and LLM-powered answers.

## Architecture

### PDF Pipeline
```
PDF → PyPDFLoader extracts text → Chunked → Embedded (all-MiniLM-L6-v2) → FAISS index
Query → Embedded (all-MiniLM-L6-v2) → FAISS retrieves top-k chunks → OpenAI LLM generates answer
```

### Image Pipeline
```
Image → CLIP (ViT-L-14) embeds directly → FAISS index
Query → CLIP embeds text → FAISS retrieves best match → OpenAI LLM describes matched image
```

## Prerequisites

- **Python 3.11**
- **Conda** (Miniforge recommended for Mac, Miniconda/Anaconda for Linux)
- **OpenAI API key** with credits loaded
- **chafa** (optional, for terminal image display)

## Setup

### Mac (Apple Silicon M1/M2/M3/M4)

> **Important:** Use [Miniforge](https://github.com/conda-forge/miniforge) (ARM-native conda), not Anaconda (x86). Anaconda limits PyTorch to v2.2.2 on Apple Silicon.

```bash
# Install Miniforge if not already installed
brew install miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
# Restart terminal after this

# Optional: install chafa for terminal image display
brew install chafa
```

### Linux (Ubuntu/Debian)

```bash
# Install Miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart terminal

# Optional: install chafa for terminal image display
sudo apt install chafa
```

### Linux (Fedora/RHEL)

```bash
# Install Miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart terminal

# Optional: install chafa for terminal image display
sudo dnf install chafa
```

---

### Common Setup (Mac & Linux)

#### 1. Create Conda Environment

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

#### 2. Install PyTorch (first, separately)

**Mac (Apple Silicon):**
```bash
pip install torch torchvision
```

Verify MPS support:
```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# Should show: 2.x.x True
```

**Linux (CPU only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Linux (with NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA support:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should show: 2.x.x True (if GPU available)
```

#### 3. Install Dependencies

```bash
pip install "numpy<2"
pip install -r requirements.txt
pip install docx2txt
pip install langchain-text-splitters
```

#### 4. Set Up API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-your_actual_key_here
```

No quotes needed around the key.

#### 5. Add Your Data

Place your files in the `data/` folder:

```
data/
├── paper1.pdf
├── paper2.pdf
├── cat.png
├── dog.jpeg
├── diagram.jpg
└── subfolder/
    └── more_files_here.pdf
```

**Supported formats:**
- **PDFs:** `.pdf`
- **Text:** `.txt`
- **CSV:** `.csv`
- **Excel:** `.xlsx`
- **Word:** `.docx`
- **Images:** `.png`, `.jpg`, `.jpeg`

## Usage

### Search PDFs

```bash
python app.py --pdfs --query "What is attention mechanism?"
```

Or simply (defaults to PDFs):
```bash
python app.py --query "What is attention mechanism?"
```

First run auto-builds the FAISS index (`faiss_store/`). Subsequent runs load it instantly.

### Search Images

```bash
python app.py --images --query "Show me the cat"
python app.py --images --query "a person in suit"
python app.py --images --query "famous tower"
```

First run downloads the CLIP model (~900MB, one time) and builds the image index (`faiss_store_images/`). Subsequent runs load from saved index.

### Rebuild Indexes

After adding new files to `data/`, rebuild the relevant index:

```bash
# Rebuild image index
python app.py --images --rebuild --query "your query"

# Rebuild PDF index (delete and re-run)
rm -rf faiss_store
python app.py --pdfs --query "your query"
```

## Project Structure

```
RAG-DistressNet/
├── app.py                  # CLI entry point (--pdfs / --images / --rebuild)
├── requirements.txt        # Python dependencies
├── .env                    # OpenAI API key (create this)
├── data/                   # Place PDFs and images here
├── faiss_store/            # Auto-generated PDF vector index
├── faiss_store_images/     # Auto-generated image vector index
└── src/
    ├── __init__.py
    ├── data_loader.py      # Loads PDFs, TXT, CSV, Excel, Word + image paths
    ├── embedding.py        # Text chunking and embedding (all-MiniLM-L6-v2)
    ├── vectorstore.py      # FAISS vector store for text documents
    ├── clip_store.py       # CLIP-based vector store for images (ViT-L-14)
    └── search.py           # RAGSearch (PDFs) + ImageRAGSearch (images)
```

## How It Works

### PDF Search
1. **Indexing:** PDFs are loaded → split into chunks (1000 chars, 200 overlap) → embedded using `all-MiniLM-L6-v2` → stored in FAISS
2. **Querying:** Query is embedded → FAISS finds top-k similar chunks → chunks sent as context to OpenAI GPT → LLM generates answer

### Image Search
1. **Indexing:** Images are embedded directly using CLIP (`ViT-L-14`) → stored in FAISS. No LLM needed at index time.
2. **Querying:** Query text is embedded by CLIP → FAISS finds best matching image(s) → matched image sent to GPT-4o-mini for description
3. **Matching:** Supports both visual matching (CLIP similarity) and filename matching (e.g., query "prof stoleru" matches `prof_stoleru.jpeg`)

### Why CLIP for Images?
CLIP understands both images and text in the same vector space. It was trained on 400M image-text pairs, so it knows that an image of a cat and the text "a photo of a cat" are similar — without needing an LLM to describe the image first. This keeps indexing fast, free, and local.

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Embeddings | `all-MiniLM-L6-v2` (~90MB) | Embeds PDF text chunks |
| Image Embeddings | `ViT-L-14` via OpenCLIP (~900MB) | Embeds images and query text |
| LLM | `gpt-4o-mini` (OpenAI API) | Generates answers from retrieved context |

## Dependencies

```
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain_openai
pypdf
pymupdf
sentence-transformers
faiss-cpu
chromadb
python-dotenv
typesense
langgraph
Pillow
open-clip-torch
docx2txt
```

## Troubleshooting

### PyTorch version stuck at 2.2.2 (Mac)
You're using Intel Anaconda on Apple Silicon. Install Miniforge:
```bash
brew install miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
# Restart terminal
```

### PyTorch not detecting GPU (Linux)
Make sure you installed the CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Check NVIDIA drivers:
```bash
nvidia-smi
```

### `ModuleNotFoundError: No module named 'langchain.text_splitter'`
```bash
pip install langchain-text-splitters
```
And update import: `from langchain_text_splitters import RecursiveCharacterTextSplitter`

### `ModuleNotFoundError: No module named 'langchain.schema'`
Use updated imports:
```python
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
```

### NumPy 2.x compatibility error
```bash
pip install "numpy<2"
```

### Segmentation fault with FAISS (Mac)
```bash
pip uninstall faiss-cpu -y
conda install -c conda-forge faiss-cpu -y
```

### Segmentation fault with FAISS (Linux)
```bash
pip uninstall faiss-cpu -y
pip install faiss-cpu --force-reinstall --no-cache-dir
```

### `open` command not found for images (Linux)
The `open` command is Mac-only. On Linux, use `xdg-open` instead. In `app.py`, change:
```python
subprocess.run(["open", img["path"]])
```
to:
```python
subprocess.run(["xdg-open", img["path"]])
```
Or use `chafa` for terminal display (works on both Mac and Linux):
```python
subprocess.run(["chafa", "--size=40x20", img["path"]])
```

### CLIP not matching "human" or "person" correctly
The ViT-L-14 model works best with descriptive queries. Try:
- `"a person in suit"` instead of `"human"`
- `"man portrait photo"` instead of `"person"`

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.