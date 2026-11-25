"""src -- RNA secondary-structure training/evaluation codebase.

Public entry points:

* :mod:`src.data`   -- canonical JSONL/CT loading, tokenization, batching, thermo caches
* :mod:`src.model`  -- RiNALMo wrapper + pair heads + StubBackbone
* :mod:`src.loss`   -- masked BCE on eligible candidate pairs
* :mod:`src.decode` -- greedy threshold decoding (0-based pairs)
* :mod:`src.metrics`-- exact + flexible F1, RNA- and family-macro
* :mod:`src.official` -- official Zenodo checkpoint loader, decoder and metric replica

Run ``python scripts/train.py --config configs/baseline_e2e.yaml`` to train and
``python scripts/evaluate.py --checkpoint <run>/best.pt --config ... --split TS0``
to evaluate.  See ``src/README.md`` for the interface contract.
"""

__version__ = "0.1.0"
