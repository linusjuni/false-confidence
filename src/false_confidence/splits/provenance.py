"""Gold (expert) vs. silver (machine-generated) label provenance.

The public MIDRC metadata marks each CSpineSeg segmentation file with an
``annotation_method``: ``Retrospective_expert`` for the 491 expert-reviewed
masks and ``Retrospective_auto`` for the rest. We query the public MIDRC
Guppy API (no login needed) and map file names to series IDs via the
``annotation_file`` metadata table.
"""

from __future__ import annotations

import json
import urllib.request

import polars as pl

from false_confidence.data.loader import structured_tsv
from false_confidence.data.schemas import Col
from false_confidence.logger import get_logger

logger = get_logger(__name__)

GUPPY_URL = "https://data.midrc.org/guppy/graphql"
PROJECT_ID = "Open-Duke-CSpineSeg"
EXPERT = "Retrospective_expert"

_QUERY = """
query ($filter: JSON) {
  data_file(first: 10000, filter: $filter) { file_name }
}
"""


def fetch_expert_seg_filenames() -> set[str]:
    """Return the file names of all expert-annotated CSpineSeg masks on MIDRC."""
    body = {
        "query": _QUERY,
        "variables": {
            "filter": {
                "AND": [
                    {"eq": {"project_id": PROJECT_ID}},
                    {"eq": {"annotation_method": EXPERT}},
                ]
            }
        },
    }
    request = urllib.request.Request(
        GUPPY_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "errors" in payload:
        raise RuntimeError(f"MIDRC query failed: {payload['errors']}")
    names = {row["file_name"] for row in payload["data"]["data_file"]}
    logger.success("Fetched expert annotations from MIDRC", n_files=len(names))
    return names


def gold_series_ids() -> set[str]:
    """Series IDs whose reference segmentation is expert-annotated (gold)."""
    seg_names = fetch_expert_seg_filenames()
    files = pl.read_csv(structured_tsv("annotation_file"), separator="\t")
    gold = files.filter(pl.col("file_name").is_in(seg_names))
    ids = set(gold["mr_series_files.submitter_id"].to_list())
    if len(ids) != len(seg_names):
        logger.warning("Some expert files did not match a series", matched=len(ids))
    return ids


def add_annotation_quality(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``annotation_quality`` (``gold`` / ``silver``) to an exam-level frame."""
    gold = gold_series_ids()
    return df.with_columns(
        pl.when(pl.col(Col.SERIES_SUBMITTER_ID).is_in(gold))
        .then(pl.lit("gold"))
        .otherwise(pl.lit("silver"))
        .alias("annotation_quality")
    )
