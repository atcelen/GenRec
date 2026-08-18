"""GenRec: Knowing Where to Reconstruct and Where to Generate."""

__version__ = "1.0.0"

__all__ = ["load_genrec", "resolve_weights"]


def __getattr__(name):
    # Lazy re-exports: keep `import genrec` light (no torch/diffusers import).
    if name == "load_genrec":
        from genrec.cli.inference import load_genrec
        return load_genrec
    if name == "resolve_weights":
        from genrec.weights import resolve_weights
        return resolve_weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
