# Assignment 2B — RAG Pipeline on *The Constitution of India (2026)*

A small but complete retrieval-augmented generation (RAG) system built on the Assignment 1 corpus:
1. Choose a chunk size.
2. Compare dense, keyword (BM25) and hybrid search.
3. Rerank the results with a cross-encoder.
4. Generate grounded answers with chunk citations.
5. Check the answers with an LLM judge.

```text
corpus_clean.txt ──fixed-size chunks──► dense (bge + FAISS) │ BM25 │ hybrid (RRF) ──► cross-encoder rerank ──► top-3 chunks
                                                                                                             │
           LLM judge (correct? supported?) ◄── grounded answer with [n] citations ◄── generator ◄────────────┘
```

## Group members

| Sr. No. | Student ID | Name |
| :---: | :--- | :--- |
| 1 | `2025ae05573` | Darshil Sanjay Shah |
| 2 | `2025ae05830` | Prashant V Panchal |
| 3 | `2025ae05839` | Jena Ajit Anirudha Ranjita |
| 4 | `2025ae05840` | Kulkarni Divya Prakash Revati |

## Repository layout

| Path | What it is |
|---|---|
| `rag_pipeline.ipynb` | **The assignment notebook** (Parts A–D). Upload this one to Colab. |
| `a1_retrain_model.ipynb` | Retrains the Assignment 1 model (the original adapters were not kept) and saves it to Google Drive. |
| `outputs/` | Executed copies of both notebooks from Colab, for submission. |
| `llm_assingment_1/` | Assignment 1 material: `corpus_clean.txt`, `sections.jsonl`, `instructions.jsonl`, and the A1 notebook. |
| `Assignment 2B — RAG Pipeline.pdf` | The assignment brief. |

## What is reused from Assignment 1

- **Documents:** `corpus_clean.txt`. These are the 518 sections of the Constitution (Preamble, articles, schedules, appendices), each with a provenance header.
- **Queries:** 10 questions from the held-out evaluation split of `instructions.jsonl`, with their expected answers. They are 6 factual, 1 procedural, 1 explanatory, 1 scenario, and 1 whose wording differs from the legal text. For each one, we marked a verbatim **answer span** in the corpus. The notebook checks that each span occurs exactly once.
- **Model:** `Qwen/Qwen3-0.6B-Base`, after continual pre-training (QLoRA r=64, 3 epochs) and SFT (QLoRA r=16, 3 epochs, 200 Q&A pairs). `a1_retrain_model.ipynb` retrains it with the same data and hyper-parameters. The retrained model matches Assignment 1:

  | Check | Retrained | Assignment 1 |
  |---|---|---|
  | Held-out perplexity (base → CPT) | 5.46 → 3.44 | 5.46 → 3.45 |
  | ROUGE-L on the 39 eval questions | 0.187 | 0.190 |
  | Token F1 on the 39 eval questions | 0.281 | 0.280 |

## How to run (Google Colab)

1. **Prepare Google Drive.** Create `MyDrive/llm_assignment_2/data/` and upload `sections.jsonl`, `instructions.jsonl` and `corpus_clean.txt` from `llm_assingment_1/`.
2. **Use a T4 GPU runtime.** Runtime → Change runtime type → T4 GPU.
3. **Run `a1_retrain_model.ipynb` once** (about 10 minutes). It writes `cpt-adapter/`, `cpt-merged/`, `sft-on-cpt-adapter/` and `a1_model_info.json` to `MyDrive/llm_assignment_2/models/`. When run again, it skips anything already on Drive. Set `FORCE_RETRAIN = True` to train again.
4. **Run `rag_pipeline.ipynb`** with Runtime → Run all (about 15–20 minutes). Most of that time is the first download of the judge model, about 29 GB for phi-4.
5. **Save the outputs.** Use File → Save, then run the last cell. It writes `rag_pipeline.html` to `MyDrive/llm_assignment_2/results/rag/`. Download the executed `.ipynb` and the `.html` for submission.

> **Make sure the downloaded notebook contains outputs.** If Colab's *Edit → Notebook settings → "Omit code cell output when saving this notebook"* is ticked, or the file is downloaded before the run finishes, the `.ipynb` will contain code only.

Optional settings:

| Setting | Where | Effect |
|---|---|---|
| Colab secret `HF_TOKEN` | 🔑 icon in Colab's left bar | Only needed if your HF account has access to the gated `meta-llama/Llama-3.1-8B-Instruct` judge. |
| `JUDGE_OVERRIDE` | Judge-selection cell | Forces a judge model, e.g. `"mistralai/Mistral-7B-Instruct-v0.3"` (smaller download). |
| `FORCE_GENERATOR` | The cell after the A1 answers | `"a1"` or `"instruct"` overrides the automatic choice of generator. |

## Design choices

| Component | Choice | Why |
|---|---|---|
| Embedder | `BAAI/bge-base-en-v1.5` (768-dim, 512-token limit), with bge's query prefix | `all-MiniLM-L6-v2` truncates at 256 tokens, which would cut off every 400- and 500-token chunk. The brief allows switching to a model with a longer limit. |
| Vector index | FAISS `IndexFlatIP` on L2-normalised vectors | Exact cosine similarity. |
| Chunking | Fixed size at **100, 400 and 500** tokens, no overlap, never cutting a word | The brief's table says 100/500 but its text says 100/400, so we ran all three. |
| BM25 | `rank_bm25.BM25Okapi`, lowercase alphanumeric tokens, short stop-word list | Article numbers such as `243zj` stay as one token. |
| Hybrid | RRF with k = 60 over the top 20 of each list | As specified in the brief. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`: top 10 → top 3 | As specified in the brief. |
| Generator | The A1 model is tried first. The notebook switches automatically to **`Qwen/Qwen2.5-7B-Instruct`** (4-bit) if the A1 model fails the instruction checks. | The A1 model invented an answer to a question the corpus cannot answer: it named a fictional party for the 2024 election. |
| Judge | The first model the account can download from: Llama-3.1-8B → **`microsoft/phi-4`** → Mistral-7B-v0.3 → Granite-3.3-8B. It must differ from the generator. | Llama is gated. phi-4 is open, comes from another model family, and fits a T4 in 4-bit. |
| Decoding | Greedy (`do_sample=False`, same as temperature 0) for the generator and the judge | Results are reproducible. |

The analysis text quotes numbers from the run. Code cells beside each analysis print **✓** if the numbers in the text match the current run, or **⚠** naming what changed.

## Results

Parts A–C gave the same numbers on Colab (T4) and in a local CPU run.

**A2: chunk size**

| Chunk size | Chunks | Hit rate@3 | Tokens sent per query (top 3) |
|---|---|---|---|
| 100 | 1537 | 80% | ≈ 300 |
| **400 (chosen)** | 384 | **90%** | ≈ 1200 |
| 500 | 308 | 80% | ≈ 1500 |

**B: search methods** (400-token chunks, top 5)

| Method | Latency | Hit rate@1 | Hit rate@3 |
|---|---|---|---|
| **Dense (chosen)** | ≈ 12 ms | 60% | **90%** |
| BM25 | < 1 ms | 60% | 80% |
| Hybrid (RRF) | ≈ 14 ms | 70% | 80% |

**C: reranking** (dense top 10 → cross-encoder → top 3)

| Measure | Value |
|---|---|
| Time per query | ≈ 53–56 ms (T4) |
| Rank change rate | 30% |
| Hit rate@1 before → after | 60% → 70% |
| Hit rate@3 before → after | 90% → 80% |

**D: generation and judging** (Colab run of 8 Nov 2026)

- **Generator:** Qwen2.5-7B-Instruct.
  - Both questions the corpus cannot answer (the 2024 election winner and the GST rate on restaurant food) got "Not found in the documents".
  - No answer was only a chunk number.
- **Judge:** `microsoft/phi-4`.

  | Judge verdict | Count out of 10 |
  |---|---|
  | Correct | 6 |
  | Partly correct | 1 |
  | Wrong | 3 |
  | Supported by the chunks | 9 |

- **Spot-check (D2.2):** we agree with all 3 verdicts we checked. For coi-067 we disagree with the judge's *reason*: it says the answer was in the chunks, but it was not. Reading all ten answers also showed two weak spots:
  - wrong chunk citations are not checked (coi-094 cites a non-existent [4]);
  - the judge is lenient on references: it marked coi-221 "supported" although the answer cites "Article 102", which is not in its chunks.
- **Failure analysis (D2.3):**
  - coi-067 is a *retrieval* failure. Dense search ranked the chunk with the 530-member cap 2nd, and reranking pushed it to 4th.
  - coi-129 is a *generation* failure. Its answer is split across two chunks.
- **Suggested fix (D2.4):** send the top 5 reranked chunks instead of the top 3. Then the answer span reaches the generator for 9 of 10 queries instead of 8. The notebook re-generates the coi-067 answer to test this. It changes from "Not found" to "not more than five hundred and thirty members … [4]", which is grounded and correctly cited. It still leaves out the 20 Union-territory members.

The executed notebook in `outputs/` (and its HTML export `outputs/rag_pipeline.html`) is the source of truth for the final numbers.

## Files written to Drive

| Path | Contents |
|---|---|
| `models/` | The retrained A1 model (`cpt-merged`, `sft-on-cpt-adapter`) and `a1_model_info.json` |
| `results/rag/rag_results.json` | All tables, every answer, the judge verdicts with raw judge output, and the fix test |
| `results/rag/judge_table.csv` | The per-query judge table |
| `results/rag/rag_pipeline.html` | The HTML export for submission |

## Local smoke test (optional, CPU)

Both notebooks have a `SMOKE_TEST=1` mode. It runs Parts A–C unchanged and uses tiny models for generation and judging, which checks that the code runs end to end. The answers in this mode are meaningless.

```bash
SMOKE_TEST=1 PROJECT_DIR=/path/to/project \
  jupyter nbconvert --to notebook --execute rag_pipeline.ipynb --output-dir /tmp/smoke
```

`PROJECT_DIR` must contain `data/` and a `models_smoke/` folder. Running `a1_retrain_model.ipynb` with `SMOKE_TEST=1` creates that folder.
