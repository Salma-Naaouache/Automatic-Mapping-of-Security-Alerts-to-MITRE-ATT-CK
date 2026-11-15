"""attackmap — automatic mapping of security alerts and CTI reports to MITRE ATT&CK.

Pipeline:
    1. ``attackmap.data``      build an ATT&CK knowledge base + a labelled corpus
                               (ATT&CK procedure examples, TRAM CTI sentences, Sigma SIEM rules)
    2. ``attackmap.models``    TF-IDF baseline, BiLSTM, BERT / RoBERTa fine-tuned classifiers,
                               Sentence-Transformer retrieval (zero-shot + fine-tuned)
    3. ``attackmap.evaluate``  ranking & multi-label metrics
    4. ``attackmap.predictor`` ensemble inference -> techniques, tactics, evidence
    5. ``attackmap.api``       FastAPI service + web console + ATT&CK Navigator export
"""

__version__ = "1.0.0"
