#!/usr/bin/env bash
# Creates the initial (empty) file tree for this project.
#
# This is a record of how the repo skeleton was generated, kept runnable so it
# can be re-applied to a fresh clone. It is idempotent: `mkdir -p` and `touch`
# leave existing directories and files untouched, so running it will never
# clobber work already in these files.
set -euo pipefail

cd "$(dirname "$0")/.."

# `touch` does not create parent directories, so they come first.
mkdir -p \
  backend/app/rag \
  backend/app/storage \
  backend/tests \
  scripts \
  docs/rag \
  data/samples

touch \
  README.md \
  ROADMAP.md \
  .env.example \
  backend/pyproject.toml \
  backend/app/rag/chunking.py \
  backend/app/rag/embeddings.py \
  backend/app/rag/retrieval.py \
  backend/app/rag/generation.py \
  backend/app/storage/supabase.py \
  backend/app/pipeline.py \
  scripts/ingest.py \
  docs/rag/embeddings.md \
  docs/rag/chunking.md \
  docs/rag/retrieval.md
