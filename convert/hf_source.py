"""Locate a Hugging Face checkpoint for the converters.

Every converter reads the same kind of directory (a HF release: one
``model.safetensors`` or a sharded pair with ``model.safetensors.index.json``),
so the lookup lives here instead of being hard-coded per script. Resolution
order, first hit wins:

  1. an explicit path -- ``--src`` on the CLI, or the model's env var
     (``MOONDREAM_SRC`` / ``VIDEOCHAT3_SRC``)
  2. the Hugging Face cache -- ``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``
  3. a plain download next to the repo -- ``models/<name>``

The default model is moondream3.1-9B-A2B; pass ``model_id=`` for another release::

    from hf_source import find_checkpoint_dir
    src = find_checkpoint_dir(model_id="qqceqqq/VideoChat3-4B",
                              env_var="VIDEOCHAT3_SRC",
                              patterns=("model.safetensors.index.json", "model.safetensors"))
"""
import glob
import os

DEFAULT_MODEL_ID = "moondream/moondream3.1-9B-A2B"
MODEL_ID = DEFAULT_MODEL_ID          # kept for the moondream converters
MODEL_FILE = "model.safetensors"

# patterns that mark a directory as a usable checkpoint
MOONDREAM_PATTERNS = (MODEL_FILE, "model.safetensors.index.json")


def _hub_dir():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _local_name(model_id):
    return model_id.split("/")[-1]


def _expand(path):
    """Accept a directory, or a file inside it (returns directories only)."""
    if not path:
        return []
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if os.path.isfile(path):
        return [os.path.dirname(path)]
    return [path]


def _qualifies(path, patterns):
    return bool(path) and os.path.isdir(path) and any(
        glob.glob(os.path.join(path, p)) for p in patterns)


def _candidates(src=None, model_id=DEFAULT_MODEL_ID, env_var=None):
    """Every plausible checkpoint directory, most explicit first (unfiltered)."""
    cands = _expand(src)
    if env_var:
        cands += _expand(os.environ.get(env_var))
    cache = os.path.join(_hub_dir(), "models--" + model_id.replace("/", "--"))
    snapshots = glob.glob(os.path.join(cache, "snapshots", "*"))
    # several snapshots may be cached -- prefer the newest
    cands += sorted(snapshots, key=os.path.getmtime, reverse=True)
    cands += [os.path.join(_repo_root(), "models", _local_name(model_id))]
    return cands


def candidate_dirs(src=None, model_id=DEFAULT_MODEL_ID, env_var=None, patterns=MOONDREAM_PATTERNS):
    """Plausible checkpoint directories that actually contain the weights."""
    return [c for c in _candidates(src, model_id, env_var) if _qualifies(c, patterns)]


def find_checkpoint_dir(src=None, model_id=DEFAULT_MODEL_ID, env_var=None,
                        patterns=MOONDREAM_PATTERNS):
    """Absolute directory holding the checkpoint, or a FileNotFoundError that
    explains how to download it."""
    cands = _candidates(src, model_id, env_var)
    for c in cands:
        if _qualifies(c, patterns):
            return c
    raise FileNotFoundError(
        f"{model_id} not found (looked for {' or '.join(patterns)}).\n"
        "Searched:\n  " + "\n  ".join(cands) +
        "\n\nDownload it first:\n"
        f"  hf download {model_id} --local-dir models/{_local_name(model_id)}\n"
        + (f"or point at an existing copy with --src <dir> / {env_var}=<dir>."
           if env_var else "or point at an existing copy with --src <dir>."))


def find_model_safetensors(src=None, model_id=DEFAULT_MODEL_ID, env_var="MOONDREAM_SRC"):
    """Path of ``model.safetensors`` (moondream converters)."""
    return os.path.join(find_checkpoint_dir(src, model_id, env_var), MODEL_FILE)


def add_src_argument(parser, model_id=DEFAULT_MODEL_ID, env_var="MOONDREAM_SRC"):
    """Register the shared ``--src`` flag on a converter's ArgumentParser."""
    parser.add_argument(
        "--src", metavar="PATH", default=None,
        help=f"directory of the {model_id} checkpoint (or one of its files); "
             f"defaults to ${env_var}, the HF cache, then models/")
    return parser
