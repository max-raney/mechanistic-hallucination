# Mechanistic-Hallucination Project

A replication and extension of CTGT’s feature-level interventions for censorship, hallucination, and deception in open-source LLMs.  

---

## Contributors

- Max Raney  
- Ruitian Yang  
- Keshav Ratra  
- Sheng Qian  

---

## 🚀 Quickstart

1. **Clone & enter**  
   ```bash
   git clone git@github.com:max-raney/mechanistic-hallucination.git
   cd mechanistic-hallucination
   ```

2. **Create & activate virtualenv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate     # Windows PowerShell
   ```

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CUDA-enabled PyTorch**  
   Depending on your GPU and driver, pick the matching CUDA version:

   - **pip** (example for CUDA 11.8):
     ```bash
     pip uninstall -y torch torchvision torchaudio
     pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu118
     ```

   - **conda** (example for CUDA 11.8):
     ```bash
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
       -c pytorch -c nvidia
     ```

5. **Install this package in editable mode**  
   ```bash
   pip install -e .
   ```

---

## 🔑 Model Access

This project uses private/open-weight models from Hugging Face. You’ll need a **read**-scoped token:

1. Sign in at https://huggingface.co → **Settings → Access Tokens** → **New token**  
2. Name it, select **read** scope, and **Generate** → **Copy** the resulting `hf_…` string  
3. In your project root:
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and set:
   ```
   HF_TOKEN=hf_<your-token-here>
   ```

---

## ⚙️ Usage

Once your `.env` is configured and the package is installed, run:

```bash
# Run via Python module
python -m mechanistic_interventions.evaluation.benchmark

# Or, if you defined a console script 'bench' in setup.py:
bench
```

This will output JSON containing:

- `model_id`  
- `load_time_sec`  
- `inference_time_sec`  
- `peak_memory_gb`  
- `tokens_per_sec`  
- `output_text`  

---

## 🧪 Testing

We use pytest for unit tests. To verify everything works:

```bash
pytest -q
```

---