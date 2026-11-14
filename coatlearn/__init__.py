"""COAT: causal discovery over unstructured data with large language models.

The public surface is intentionally small::

    from coatlearn import ModelConfig, RunConfig, run_mb, run_pag, run_adj

``run_mb`` / ``run_pag`` / ``run_adj`` wrap the core experiment classes
(``COATMB`` / ``COATPAG`` / ``COATADJ``) without re-implementing any
algorithm.  They validate inputs first, then resolve model clients, and
return a strictly JSON-safe result dictionary with a conservative
``status`` field.

Importing this package has no side effects: no model client is
constructed, no credentials are read, no experiment object is created,
no NLTK corpus is downloaded, and no dataset is accessed.  Heavy
dependencies (causal-learn and friends) are imported only when a ``run_*``
function is actually called.

Independent model, annotation and causal operations are available from their
functional subpackages. Importing those interfaces starts no model inference.
"""

from coatlearn.models.openai import ModelConfig
from .api import RunConfig, run_mb, run_pag, run_adj

__all__ = ["ModelConfig", "RunConfig", "run_mb", "run_pag", "run_adj"]
