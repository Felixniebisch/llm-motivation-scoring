Recognizing Intrinsic Motivation through LLM Processing

Can large language models infer intrinsic motivation from natural language?

This project investigates whether an LLM can estimate intrinsic motivation from open-ended descriptions of participants’ activities. LLM-derived scores are compared against self-reported scores from the Intrinsic Motivation Inventory (IMI), a validated measure grounded in Self-Determination Theory.

The project combines LLM-based text analysis, prompt engineering, psychometric measurement, and statistical validation.

⸻

Research question

Can an LLM infer psychological constructs that are typically measured through standardized self-report questionnaires?

More specifically:

To what extent can intrinsic motivation be estimated from free-text descriptions of an activity using an LLM?

Rather than treating the LLM output as a standalone classification, the project evaluates its predictions against participant-level psychometric measurements.

⸻

Method

The analysis follows a simple pipeline:

Participant
    │
    ▼
Open-ended activity description
    │
    ▼
LLM-based scoring
    │
    ├── Interest / Enjoyment
    ├── Perceived Competence
    ├── Effort
    ├── Value / Usefulness
    ├── Pressure / Tension
    └── Perceived Choice
    │
    ▼
Predicted IMI scores (1–7)
    │
    ▼
Comparison with participant self-report
    │
    ├── Spearman's ρ
    ├── Mean Absolute Error
    └── Prediction bias

Model

* Model: GPT-4-turbo
* Interface: OpenAI API
* Input: Open-ended descriptions of participants’ activities
* Output: Predicted Likert-scale scores (1–7)
* Prompting: Structured prompt design with reasoning-based scoring
* Validation: Comparison against IMI self-report scores

Participants

N = 245 participants

Each participant completed the IMI alongside an open-ended description of a personally relevant activity.

The analysis focuses on six IMI subscales:

1. Interest / Enjoyment
2. Perceived Competence
3. Effort
4. Value / Usefulness
5. Pressure / Tension
6. Perceived Choice

⸻

Results

LLM predictions showed varying degrees of correspondence with participant self-reports across the six motivational dimensions.

IMI subscale	Spearman’s ρ	MAE
Interest / Enjoyment	0.80	0.77
Value / Usefulness	0.62	0.77
Perceived Choice	0.57	0.97
Perceived Competence	0.50	0.68
Effort	0.43	0.87
Pressure / Tension	0.34	1.24

The strongest correspondence was observed for Interest / Enjoyment, while Pressure / Tension showed the weakest association and highest prediction error.

These differences suggest that some motivational dimensions may be more readily expressed and inferred from natural language than others.

⸻

Repository structure

File	Purpose
main.py	Main analysis loop; sends participant responses to the LLM and stores predictions
initialization_prompt.py	Defines the structured initialization/system prompt
reasoning_prompts.py	Contains scoring prompts used for the IMI dimensions
compute_averages.py	Aggregates item-level predictions into subscale scores
reversed_scales.py	Handles reverse-coded IMI items
PydanticClasses.py	Validates structured LLM responses

⸻

Installation

Requires Python 3.11+.

pip install openai pandas numpy scipy statsmodels pydantic python-dotenv

Set your OpenAI API key as an environment variable:

OPENAI_API_KEY=your_api_key

⸻

Research context

The project is based on Self-Determination Theory and the Intrinsic Motivation Inventory (IMI).

The broader research question is whether psychological constructs traditionally assessed through questionnaires can also be estimated from naturally occurring language.

This project formed part of my research on LLM-based psychometric assessment.

Related work

A manuscript based on this research has been submitted to:

IEEE Transactions on Affective Computing

⸻

Limitations

LLM-derived scores should not be interpreted as replacements for validated psychological measurement.

The objective of the project is instead to investigate:

* whether psychological constructs leave detectable traces in language,
* which constructs are more readily inferred from free text,
* and how closely LLM-based estimates correspond to established psychometric measures.

The results therefore reflect both the information contained in the participants’ language and the behavior of the particular LLM and prompting strategy used.

⸻

Citation

If you use this code or methodology, please cite the associated research publication.

Felix Niebisch & Loïs Vanhee
Automatically Identifying Motivation From Text

[Add publication / preprint link]
