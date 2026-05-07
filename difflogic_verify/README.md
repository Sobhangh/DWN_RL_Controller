# Logic Gate Neural Networks for Verification  
_Official code for the paper “Logic Gate Neural Networks are Good for Verification” (NeuS ’25)_

---

## 1 · Installation

Python 3.9 is required.

```bash
bash setup.sh
```
```bash
source difflogic_verification/bin/activate
```
---

## 2 · Training Networks

```bash
python train_model.py --k <gates-per-layer> \
                      --experiment <dataset> \
                      --seed <int>
```

* **k** – number of logic gates **per layer** (all models use **3 layers**)  
* **experiment** – one of `adult`, `german_credit`, `folktable_5`, `law`, `compas`  
* **seed** – random-seed (must be reused for verification)

Trained weights are saved to **`models/`** with a self-describing filename.

### Quick example (Adult dataset)

```bash
python train_model.py --k 100 --experiment adult --seed 42
```

---

## 3 · Verifying Networks


### Verifying the *Adult* model trained above

Try multiple confidence values or sensitive attributes (`race`, `gender`, `age` (german_credit only))! Keep in mind that only confidence values [1/classes, 1.0] make sense. Or don't supply a sensitive attribute and check global robustness by modifying epsilon.

With SMT backend (for counter-examples):

```bash
python verify_model.py \
  --experiment=adult \
  --verify=global_robustness \
  --k=100 \
  --model_path=models/model_adult_100_42.pth \
  --seed=42 \
  --confidence=0.6 \
  --epsilon=0 \
  --usesmt \
  --sensitive_attribute=gender

```
with kissat:
```bash
python verify_model.py \
  --experiment=adult \
  --verify=global_robustness \
  --k=100 \
  --model_path=models/model_adult_100_42.pth \
  --seed=42 \
  --confidence=0.6 \
  --epsilon=0 \
  --sensitive_attribute=gender
```

---


```bash
python verify_model.py --experiment <dataset> \
                 --verify <test_acc|global_robustness|local_robustness> \
                 --model_path <path/to/model.pth> \
                 --seed <int> --k <int> \
                 [--confidence <float>] [--epsilon <int>] \
                 [--epsilon_sensitive <int>] \
                 [--sensitive_attribute <str>] \
                 [--usesmt] [--cnf <file>] \
                 [--output <dir>]
```

| Flag | Purpose |
|------|---------|
| `--experiment` | Dataset name (must match training) |
| `--verify` | Goal: **`test_acc`**, **`global_robustness`**, **`local_robustness`** |
| `--confidence` | κ-confidence for global robustness \[0 … 1] |
| `--epsilon` | Perturbation radius for global checks |
| `--epsilon_sensitive` | Extra ε for German-Credit age (default = 1) |
| `--sensitive_attribute` | Name of sensitive feature to analyse |
| `--usesmt` | Switch to Z3-SMT backend (produces counter-examples) |
| `--cnf` | Dump DIMACS CNF instead of solving |
| `--model_path` | Path to the trained `.pth` file |
| `--output` | Directory for logs/results (default =`results`) |
| `--seed`, `--k` | **Must match training run** |

**Default solver:** Kissat.  
Add `--usesmt` to obtain counter-examples.




## 5 · Citation

```bibtex
@inproceedings{kresse2025logicgates,
  title     = {Logic Gate Neural Networks are Good for Verification},
  author    = {Kresse, Fabian and Yu, Emily and Lampert, Christoph~H. and Henzinger, Thomas~A.},
  booktitle = {Proceedings of the 2nd International Conference on Neuro-symbolic Systems (NeuS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {288},
  year      = {2025},
  publisher = {PMLR},
  url       = {https://github.com/HyberionBrew/difflogic_verify},
  note      = {Code available at \url{https://github.com/HyberionBrew/difflogic_verify}}
}
```

---

## 6 · License

Released under the **MIT License**; see [`LICENSE`](./LICENSE) for details.  
Dataset files retain their original licenses.
