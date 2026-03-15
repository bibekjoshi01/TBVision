from langchain_core.prompts import PromptTemplate  # type: ignore[import]

explanation_prompt = PromptTemplate(
    input_variables=[
        "prediction_label",
        "prob",
        "region",
        "patient_info",
    ],
    template="""You are a medical AI validator. Analyze this chest X-ray and provide a concise validation report.

CNN Assessment: {prediction_label} ({prob:.1%} probability)
CNN Attention: {region}
Patient Metadata: {patient_info}

Provide a brief assessment (3-4 sentences):
1. Do you see findings consistent with TB?
2. Does the CNN attention region make sense?
3. Any concerns or alternative diagnoses?
4. Agreement level with CNN (Agree/Partially Agree/Disagree)""",
)


synthesis_prompt = PromptTemplate(
    input_variables=[
        "prediction_label",
        "prob",
        "uncertainty",
        "uncertainty_std",
        "region",
        "gemini_validation",
        "patient_info",
        "evidence_text",
    ],
    template="""
You are a senior TB clinical decision support specialist. Synthesize all data into ONE comprehensive clinical report.

PATIENT INFORMATION
{patient_info}

DATA SOURCES

1. CNN Deep Learning Model
- Prediction: {prediction_label}
- TB Probability: {prob}
- Uncertainty Level: {uncertainty}
- Uncertainty STD: {uncertainty_std}
- Grad-CAM Attention Region: {region}

2. Gemini 2.5 Flash Independent Validation
{gemini_validation}

2. WHO Evidence (Ground Truth Knowledge)
{evidence_text}

YOUR TASK

Provide a comprehensive clinical synthesis with the following sections.

---

## Recommendation

Based on WHO TB screening guidelines, provide clear next steps.

- For positive screens: recommend confirmatory tests (sputum microscopy, culture, or GeneXpert).
- If uncertainty exists: recommend further imaging or clinical evaluation.
- Monitor symptoms for suspected but uncertain cases.
- Flag urgent cases requiring immediate referral.
- Consider age-specific TB risk and symptom patterns.

---

## Radiographic Assessment

- Summarize findings from the CNN model and Gemini validation.
- Note agreement or disagreement between AI systems.
- Evaluate whether Grad-CAM attention corresponds with potential pathology regions.
- Discuss possible radiographic patterns (infiltrates, cavitation, consolidation).
- Comment on possible imaging limitations.

---

## Clinical Correlation

- Integrate symptoms with radiographic findings.
- Evaluate whether symptoms support TB suspicion.
- Consider patient age and regional TB prevalence.
- Discuss possible differential diagnoses such as:
  - Pneumonia
  - Lung cancer
  - Non-tuberculous infections
  - Normal variation

---

## Limitations & Uncertainty

- Interpret CNN uncertainty and its clinical significance.
- Mention possible AI limitations or biases.
- Highlight disagreements between CNN and Gemini if present.
- Note possible technical or imaging quality limitations.

---

## Evidence-Based Context

- Reference WHO tuberculosis screening guidelines.
- Support recommendations with retrieved medical evidence.
- Cite relevant clinical practices where applicable.

---

Write the report in clear clinical language suitable for clinicians.
This is the final report that will be presented to medical staff.
""",
)
