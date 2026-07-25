# Data guide

No dataset files are stored in this repository. They are downloaded and built
locally when needed.

## Files created by the pipeline

`python -m src.cli download` creates:

```text
data/raw/userid_trackid_count.tsv.bz2   # user, track, and play count (~550 MB)
data/raw/id_genres_tf-idf.tsv.bz2       # track genre features (~3 MB)
```

`python -m src.cli prepare` then creates:

```text
data/prepared/interactions.parquet      # selected interactions
data/prepared/catalog.json              # fixed 10,000-track catalog
data/prepared/user_assignments.json     # training and evaluation users
data/prepared/dataset_summary.json      # counts quoted in the report
```

Both directories are ignored by Git. Delete them at any time; `python -m
src.cli all` recreates everything.

The source is the official
[Music4All-Onion v2 Zenodo record](https://zenodo.org/records/15394646).
