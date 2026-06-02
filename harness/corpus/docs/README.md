# RAG corpus documents

Drop guideline documents here as **plain-text `.txt`** files, one source per
file. The filename (minus `.txt`) becomes the `source` label shown in citations.

Suggested sources (convert the relevant sections to .txt):
- `usda_dga_2020_2025.txt`   — USDA Dietary Guidelines for Americans
- `who_healthy_diet.txt`     — WHO healthy-diet guidance (sugars, sodium, fats)
- `who_physical_activity.txt`— WHO 2020 Physical Activity Guidelines
- `acsm_exercise.txt`        — ACSM exercise prescription summary
- `issn_position_stands.txt` — protein, nutrient timing, creatine, caffeine
- `ada_diabetes.txt`, `acog_pregnancy.txt`, `kdoqi_ckd.txt` — clinical summaries

Then build the index:
    python3 corpus/build_index.py

`a_sample.txt` is included only so the pipeline runs before you add real docs.
Replace it with real guideline text for the actual study, and cite the sources.
"""
