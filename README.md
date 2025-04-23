# Mechanistic-Hallucination Project

A replication and extension of CTGT’s feature-level interventions for censorship, hallucination, and deception in open-source LLMs.  

---

## Contributors

- Max Raney  
- Ruitian Yang  
- Keshav Ratra  
- Sheng Qian  

---

## Quickstart

1. **Clone & enter**  
   ```bash
   git clone git@github.com:max-raney/mechanistic-hallucination.git
   cd mechanistic-hallucination
   ```

2. **Create & activate virtualenv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows PowerShell
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CUDA-enabled PyTorch**  
   - **pip** (CUDA 11.8 example)  
     ```bash
     pip uninstall -y torch torchvision torchaudio
     pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu118
     ```  
   - **conda** (CUDA 11.8 example)  
     ```bash
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
       -c pytorch -c nvidia
     ```

5. **Install in editable mode**  
   ```bash
   pip install -e .
   ```

---

## Model Access

1. Go to https://huggingface.co → **Settings → Access Tokens** → **New token**  
2. Name it, select **read** scope, **Generate**, copy the `hf_…` string  
3. In project root:  
   ```bash
   cp .env.example .env
   ```  
   Edit `.env` to include:  
   ```
   HF_TOKEN=hf_<your-token-here>
   ```

---

## Benchmark Usage

```bash
python -m mechanistic_interventions.evaluation.benchmark \
  [--models MODEL_ID [MODEL_ID ...]] \
  [--prompt "Your prompt"] \
  [--output PATH/TO/results.json]
```

- `--models`: list of HF model IDs (default: `google/gemma-2b meta-llama/Meta-Llama-3-8B`)  
- `--prompt`: text prompt (default: `"Hello, my name is"`)  
- `--output`: output JSON file path (default: `benchmark_results.json`)  

### Quick-exit (CI/tests)

```bash
export MECH_QUICK=1      # Linux/macOS/Git Bash
$Env:MECH_QUICK = '1'    # Windows PowerShell
set MECH_QUICK=1         # Windows CMD

python -m mechanistic_interventions.evaluation.benchmark
# prints: []
```

---

## 🧪 Testing

```bash
pytest -q
```

---