"""Publish a converted GGUF (+ its model card) to Hugging Face and ModelScope.

Both hubs are authenticated from the environment, so no token ever lands on a
command line:

    $env:HF_TOKEN = "hf_..."                # huggingface.co/settings/tokens
    $env:MODELSCOPE_API_TOKEN = "..."       # modelscope.cn -> 访问令牌, or `modelscope login`

    python scripts/upload_gguf.py --file videojev-4b-q8_0.gguf \\
        --card scripts/cards/VideoJev-4B-GGUF.md \\
        --ms-card scripts/cards/VideoJev-4B-GGUF.ms.md \\
        --hf-repo  SkyContact/VideoJev-4B-GGUF \\
        --ms-repo  SkyContact/VideoJev-4B-GGUF \\
        --remote-name VideoJev-4B-q8_0.gguf --commit-message "VideoJev-4B q8_0"

Pass ``--hf`` / ``--ms`` to restrict targets, and ``--dry-run`` to only print
what would happen (plus the auth identity each side resolves to).
"""
import argparse
import os
import sys


def _hf_stored_token():
    """Token saved by `hf auth login` (~/.cache/huggingface/token)."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def hf_upload(path, card, repo, remote_name, message, dry):
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"[hf] auth: {who.get('name')}", flush=True)
    if dry:
        print(f"[hf] would create {repo} and upload {path} -> {remote_name}")
        return
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=remote_name, repo_id=repo,
                    commit_message=message)
    if card and os.path.isfile(card):
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo,
                        commit_message="model card")
    print(f"[hf] done: https://huggingface.co/{repo}", flush=True)


def ms_upload(path, card, repo, remote_name, message, dry, base_model="", chinese_name="",
              license_name="Apache License 2.0"):
    from modelscope.hub.api import HubApi

    token = os.environ.get("MODELSCOPE_API_TOKEN") or None
    api = HubApi()
    try:
        info = api.get_model(repo, revision="master", token=token)
        print(f"[ms] repo exists: {getattr(info, 'Name', repo)}", flush=True)
        exists = True
    except Exception:
        exists = False
    if dry:
        print(f"[ms] would {'upload into' if exists else 'create'} {repo} "
              f"({path} -> {remote_name})")
        return
    if not exists:
        # original_model_id is what ModelScope turns into the lineage edge; it has to
        # be a repository that exists on *this* hub, otherwise the edge is dropped
        api.create_model(repo, visibility=5, license=license_name,
                         chinese_name=chinese_name or repo,
                         original_model_id=base_model, token=token)
    # note: the SDK takes repo_id (model_id is only used by create_model)
    api.upload_file(path_or_fileobj=path, path_in_repo=remote_name, repo_id=repo,
                    token=token, repo_type="model", commit_message=message)
    if card and os.path.isfile(card):
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo,
                        token=token, repo_type="model", commit_message="model card")
    print(f"[ms] done: https://modelscope.cn/models/{repo}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True, help="local .gguf")
    ap.add_argument("--card", default="", help="markdown model card -> README.md")
    ap.add_argument("--ms-card", default="",
                    help="ModelScope card (defaults to --card); the two hubs want "
                         "different relative links, hence the separate file")
    ap.add_argument("--license", default="Apache License 2.0",
                    help="ModelScope license label used when creating the repo")
    ap.add_argument("--hf-repo", default="", help="e.g. SkyContact/VideoJev-4B-GGUF")
    ap.add_argument("--ms-repo", default="", help="e.g. SkyContact/VideoJev-4B-GGUF")
    ap.add_argument("--remote-name", default="", help="file name inside the repo")
    ap.add_argument("--commit-message", default="", help="commit message")
    ap.add_argument("--hf", action="store_true", help="only Hugging Face")
    ap.add_argument("--ms", action="store_true", help="only ModelScope")
    ap.add_argument("--base-model", default="",
                    help="ModelScope original_model_id (lineage); must exist on ModelScope")
    ap.add_argument("--chinese-name", default="", help="ModelScope display name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    remote = args.remote_name or os.path.basename(args.file)
    msg = args.commit_message or f"add {remote}"
    if not os.path.isfile(args.file):
        print(f"missing {args.file}", file=sys.stderr)
        return 2
    both = not (args.hf or args.ms)
    rc = 0
    if (both or args.hf) and args.hf_repo:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or _hf_stored_token()):
            print("[hf] no token - run `hf auth login` or export HF_TOKEN", file=sys.stderr)
            rc = 1
        else:
            hf_upload(args.file, args.card, args.hf_repo, remote, msg, args.dry_run)
    if (both or args.ms) and args.ms_repo:
        try:
            ms_upload(args.file, args.ms_card or args.card, args.ms_repo, remote, msg,
                      args.dry_run, base_model=args.base_model,
                      chinese_name=args.chinese_name, license_name=args.license)
        except Exception as e:
            print(f"[ms] failed: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
