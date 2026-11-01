# Rule

### Competition rounds

The competition runs on a leaderboard with two rounds:

- **Round 1 — Public Test:** Teams develop their solutions, run them on `public_test` and submit the results. Scores appear on the public leaderboard and are for reference only during development.
- **Final solution submission:** The final solution is submitted as a Docker image, following the organisers’ separate guidelines. For the private test, the organisers start the container at `/data`, run the inference script and collect the result at `result/submission.jsonl`. The container must run fully offline, with no network access.
- **Round 2 — Private Test:** After teams submit and freeze their solutions, the organisers prepare the private test and its environment. Teams run their frozen solution with the provided `zaic_tools` and submit the results. The organisers re-run the submitted Docker image to verify the results, and the final ranking is based on the private_test score.

### Technical requirements

Every solution must meet all of the following requirements:

- **Model size:** The whole system (ASR, LLM and any other components) must have fewer than 6B parameters in total. The LLM must have fewer than 5B parameters and a context length of at most 16K (16,384) tokens.
- **Data access:** The agent must not read `state.json` directly; all reads and writes go through the execution environment.
- **Allowed data and resources:** Teams may use public data and open-source models released before 26 October 2026, provided their licences allow it. Synthetic data may be generated with other services or models (LLM, TTS, …) during development, but these must not be used at inference. Labelling public or private test (manually or with external services or models) to train or tune a solution is not allowed.
- **Runtime and hardware:** Each sample must be processed within 30 seconds; samples that time out or fail to execute score 0. The solution must run fully offline, with no external APIs or services at inference. Scoring hardware: 1 NVIDIA A30 GPU, a 16-core Intel Xeon Gold 6442Y CPU and 64 GB RAM.
- **Reproducibility:** Fix all random seeds (Python, NumPy, PyTorch, …) and use deterministic decoding for ASR and LLM, such as greedy decoding or temperature = 0. If a re-run differs from the submitted results, the organisers run the Docker image 5 times and use the average score as the official score.
- **The solution/agent must not read, copy or parse any file under `<split>_environment/` (including `state.json`); all app data is accessed only through the organisers’ `zaic_tools`, with exactly one `ToolEnv` per sample, and every call made on it must appear in the submission.**
- **Tie-breaking:** If teams have the same final score, the organisers re-score their `tool_calls` with the F1 formula, ignoring the `state_hash` check. If they are still tied, the team with the lower average processing time per sample ranks higher.
