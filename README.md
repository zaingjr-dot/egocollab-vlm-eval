# egocollab-vlm-eval

**An exploratory evaluation of small open vision–language models (VLMs) on egocentric procedural video: can they notice mistakes, and does a task description help?**

Author: Muhammad Zain (M2 Artificial Intelligence for One Health, Université Grenoble Alpes)

> **Status: exploratory study.** 22 time windows from a single video, two small models, one wording per prompt type. The results describe how these models behaved in this setup. They do not measure how well VLMs detect mistakes in general.

## Key findings

1. **Neither model flagged a mistake.** LLaVA-OneVision 0.5B and Qwen2.5-VL 3B answered `RIGHT` for all 22 windows in every condition (0 of 11 labeled mistakes caught).
2. **An apparent signal was an artifact.** On the first test set, the 0.5B model's `WRONG`-vs-`RIGHT` score separated mistakes from correct actions with AUC ≈ 0.67–0.71. A baseline that uses only the window's *start time* reached 0.67 as well. After matching correct windows to mistake windows by position and duration, the model's AUC fell to 0.47 (chance level).
3. **Adding the task description did not help.** It changed the wording of free-form answers and made the 0.5B model slightly *more* confident in `RIGHT` (scores about 0.8 lower in the original set), but it did not make any model flag a mistake.
4. **Free-form guidance was generic and often off-topic** (invented objects, an unsupported brand name, advice that fits any camera video). See [Guidance check](#guidance-check-qualitative).

## Motivation

Giving useful guidance during a manual task requires more than describing what is visible: a model has to relate the action to the task and notice when something goes wrong. This project tests that on a small scale, using [HoloAssist](https://holoassist.github.io/), a dataset of first-person recordings where an instructor guides a performer through a task. It was built as a small study of VLMs for egocentric remote assistance.

## Data

**HoloAssist** (Wang et al., ICCV 2023). The dataset is not redistributed in this repository; download it from the official page.

- Annotation file used: `data-annotation-trainval-v1_1.json` (1,758 annotated videos, 20 tasks).
- Videos: the compressed version (`video_compress.tar`, ≈145 GB in one archive). I streamed only the first part of the archive and extracted three videos: `R007-7July-DSLR`, `R012-7July-Nespresso`, `R014-7July-DSLR`. **Only `R014-7July-DSLR` (task "setup camera", 354 s, 454 × 256 px) is used in the test sets**, because it contains most of the labeled mistakes among the three.
- Annotation statistics (fine-grained actions across the whole file, 149,253 in total):

| Label | Count | Share |
|---|---|---|
| Correct Action | 141,691 | ≈ 94.9 % |
| Wrong Action, corrected by instructor verbally | 4,052 | ≈ 2.7 % |
| Wrong Action, corrected by student | 3,061 | ≈ 2.1 % |
| otherwise | 357 | ≈ 0.2 % |
| Wrong Action, not corrected | 92 | ≈ 0.06 % |

  Mistakes are rare (≈ 5 %), so accuracy alone is misleading: a model that always answers "no mistake" would score ≈ 95 % while catching none. I therefore report precision, recall and AUC, and the test sets are balanced 50/50 on purpose (which does not reflect real mistake frequency).
- Note: the real file uses slightly different label names from the official documentation (for example "corrected by student", "otherwise").

## Method

### Test windows

Windows come from the fine-grained action annotations of `R014-7July-DSLR`.

- **Mistake windows:** actions labeled with one of the three "Wrong Action" types (11 windows).
- **Correct windows:** actions labeled "Correct Action", not overlapping any mistake window.
- Excluded: windows shorter than 1.5 s, and the ambiguous label `otherwise`.
- **Original set** (`test_set.csv`): 11 mistake windows + 11 correct windows drawn at random from 42 candidates (seed 0). The two groups turned out to differ in position and length (mean start 137.6 s vs 192.6 s; mean duration 3.3 s vs 5.7 s).
- **Matched set** (`test_set_matched.csv`): each mistake window is paired, greedily and without replacement, with the correct window minimizing `|Δstart| + 10·|Δduration|`. Group means became similar (start 137.6 s vs 139.4 s; duration 3.3 s vs 3.0 s).

### Models

| Model | Hugging Face id |
|---|---|
| LLaVA-OneVision 0.5B | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| Qwen2.5-VL 3B Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` |

Both ran in half precision on a free Colab T4 GPU through Hugging Face Transformers (PyTorch), with greedy decoding (`do_sample=False`). The 0.5B and 3B models differ in both size and family, so differences between them cannot be attributed to size alone.

### Input and conditions

Each window is shown as **4 evenly spaced frames** (`np.linspace` over the window), passed as separate images. The model never sees the audio, the instructor's speech or any label.

- **Condition A:** frames + generic question.
- **Condition B:** the same frames and question, preceded by `The person is performing this task: setup camera.` (the task name comes from the annotation file).
- Condition C (adding the annotated previous step) was **not run**.

Model, frames and generation settings are identical across conditions; only the prompt text changes.

### Prompts

- Strict: `Is the person doing this step right or wrong? Reply with exactly one word: RIGHT or WRONG.`
- Reversed order: `Is the person doing this step wrong or right? Reply with exactly one word: WRONG or RIGHT.`
- Guidance (open): `What is the person doing, is anything going wrong, and what should they do next? Answer in at most three sentences.`

An earlier free-form version ("Is the person making a mistake? Answer yes or no first…") produced self-contradictory answers (for example "No, the person is making a mistake"), so I switched to a forced one-word answer that can be scored automatically. `RIGHT`/`WRONG` were chosen because neither word contains the other, unlike `CORRECT`/`INCORRECT`.

### Metrics

- **Verdict:** the generated word, parsed as `wrong`, `right` or `unclear`. "Mistake caught" means the verdict is `wrong` on a mistake window. Precision, recall, TP/FN/FP/TN are computed with `wrong` as the positive class.
- **Score and AUC:** at the first generated position I read the model's score for the first token of `WRONG` minus its score for the first token of `RIGHT`. A positive value means the model leans toward `WRONG`. AUC measures how well this score ranks mistake windows above correct windows (0.5 = chance). With 11 vs 11 windows, the standard error of AUC under chance is about 0.13, so values between roughly 0.35 and 0.65 cannot be distinguished from chance.
- **Baselines without AI:** AUC of the window's start time and of its duration. These are reported in whichever direction works better (a rule may guess either way), while the model's AUC is not flipped.

## Results

### One-word verdicts

| Model | Test set | Prompt | Condition | Said `WRONG` | Mistakes caught |
|---|---|---|---|---|---|
| LLaVA-OV 0.5B | original | right or wrong | A | 0 / 22 | 0 / 11 |
| LLaVA-OV 0.5B | original | right or wrong | B | 0 / 22 | 0 / 11 |
| LLaVA-OV 0.5B | original | wrong or right | A | 0 / 22 | 0 / 11 |
| LLaVA-OV 0.5B | original | wrong or right | B | 0 / 22 | 0 / 11 |
| Qwen2.5-VL 3B | matched | right or wrong | A | 0 / 22 | 0 / 11 |
| Qwen2.5-VL 3B | matched | right or wrong | B | 0 / 22 | 0 / 11 |

Reversing the option order did not change the behavior, so the result is not a simple "first option" preference.

### Score-based ranking (AUC)

| Model | Test set | Condition | Model AUC | Start-time baseline | Duration baseline |
|---|---|---|---|---|---|
| LLaVA-OV 0.5B | original | A | 0.67 | 0.67 | 0.57 |
| LLaVA-OV 0.5B | original | B | 0.71 | 0.67 | 0.57 |
| LLaVA-OV 0.5B | matched | A | 0.47 | 0.51 | 0.55 |
| Qwen2.5-VL 3B | matched | A | 0.57 | 0.51 | 0.55 |
| Qwen2.5-VL 3B | matched | B | 0.52 | 0.51 | 0.55 |

On the original set, the 0.5B model's ranking tracked where a window sits in the video (every window starting before ≈ 200 s scored higher than every later one, by a tiny margin). That is why the original-set AUC should not be read as mistake detection. On the matched set, where position and duration no longer differ between groups, no model is distinguishable from chance.

### Guidance check (qualitative)

For 6 windows (3 mistake, 3 correct; seed 0) from the matched set, Qwen2.5-VL 3B answered the open guidance question under conditions A and B (12 answers, in `results/guidance_qwen3b.csv`). **These answers have not been rated by humans**; the observations below come from reading the text against the annotated action names and are not a measurement.

- Descriptions sometimes contain objects unrelated to a camera setup (a box, a printer cartridge, a card, "moving around the room"), and one answer names a brand ("Canon lens") that is probably not readable at this resolution.
- Condition B pulls descriptions toward "setting up a camera", which fits the task but is generic.
- None of the 12 answers mentions the annotated mistake (dropping the lens, pulling the battery, withdrawing the lens cover). Three answers raise an issue: one on a mistake window (for an unrelated reason) and two on correct windows.
- The advice ("hold the camera steady", "adjust the position") would fit almost any camera video and does not resemble the instructor's actual corrections.

## Limitations

- **One video, 22 windows.** Small samples make AUC very noisy, and the results may not transfer to other videos, tasks or people.
- **Low resolution.** The compressed videos are 454 × 256 px. Differences such as "lens" vs "lens cover" may not be visible even to a human; the annotators worked with the full recordings and audio.
- **Four frames per window**, sampled sparsely (≈ 0.5–1 s apart). Fast motions can fall between frames. The frame count was not varied, and video-native input (which Qwen2.5-VL supports) was not tested.
- **Two small models**, differing in size and family. Larger models and fine-tuned models were not tested.
- **One wording per prompt type.** Other prompts might behave differently.
- **Mistake labels are human judgments** and can be subjective; ambiguous `otherwise` labels were excluded.
- **Matching is heuristic** (greedy, with a hand-chosen weight). Other confounds could remain.
- **Balanced test sets** (50 % mistakes) do not reflect the real ≈ 5 % mistake rate.
- **Not done:** condition C (previous-step context), a CLIP baseline, action-recognition evaluation with a fixed label set, human ratings of guidance, frame-count and image-quality robustness tests.

## Repository contents

```
notebooks/   exploration log (see note below)
results/
  test_set.csv                 original test set (11 mistake + 11 correct windows, R014)
  test_set_matched.csv         matched test set
  results_strict.csv           LLaVA-OV 0.5B, "right or wrong", original set
  results_reversed.csv         LLaVA-OV 0.5B, "wrong or right", original set
  scores_logit.csv             LLaVA-OV 0.5B, WRONG-vs-RIGHT scores, original set
  scores_matched.csv           LLaVA-OV 0.5B, scores, matched set (condition A)
  results_qwen3b_matched.csv   Qwen2.5-VL 3B, verdicts and scores, matched set
  guidance_qwen3b.csv          open-ended guidance answers (rating columns left empty)
  summary_words.csv            summary of the one-word experiments
  summary_auc.csv              summary of AUC and baselines
```

**About the notebook:** it was run interactively in Google Colab, including session restarts, so the cells are an *exploration log* in the order the experiments were done, not a clean top-to-bottom pipeline. The results files above are the saved outputs of those runs.

## Reproducing

1. Download the annotation file and videos from the [official HoloAssist page](https://holoassist.github.io/). The compressed videos are one large archive, so extract only the sessions you need (this study used `R014-7July-DSLR`).
2. Install `transformers`, `torch`, `opencv-python`, `pandas`, `scikit-learn`, `pillow` (versions were not pinned; Colab defaults at the time of the runs, October 2026).
3. Rebuild the windows from the annotations using the rules above (seed 0), or use the provided test-set CSVs.
4. Run each model on the windows with the prompts above and greedy decoding.

## References

- Wang, X., Kwon, T., Rad, M., et al. (2023). *HoloAssist: an egocentric human interaction dataset for interactive AI assistants in the real world.* ICCV 2023.
- LLaVA-OneVision: [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Qwen2.5-VL: [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
