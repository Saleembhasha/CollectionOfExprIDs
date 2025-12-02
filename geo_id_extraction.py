import os
import requests
import pandas as pd

# ---------- 1. GenAI helper ----------


def expand_query_with_llm(user_query: str) -> str:
    """
    Take a natural-language query and turn it into a GEO DataSets esearch term.

    Replace the body with a call to your GenAI provider (OpenAI, etc.).
    For now, we just return the raw query as a fallback.
    """
    # This implementation will, when available, call the OpenAI Responses API
    # to rewrite the user's natural-language query into a single NCBI esearch
    # term. If `OPENAI_API_KEY` is not set or the call fails, we fall back to
    # a local heuristic sanitizer that removes commas and maps common
    # organism names into NCBI-style organism filters.
    import re

    openai_key = os.environ.get("OPENAI_API_KEY")

    prompt = (
        "You build NCBI GEO DataSets (db=gds) esearch terms.\n"
        "Convert this user query into a single esearch term that:\n"
        " - will be used with db=gds\n"
        " - restricts to Series records (the caller will add 'AND gse[ETYP]')\n"
        ' - includes organism if present using the format: "Homo sapiens[Organism]"\n'
        " - prefer boolean operators AND/OR and use quotes for phrases\n"
        "Return only the final esearch term, nothing else.\n\n"
        f"Query: {user_query!r}\n"
    )

    def _heuristic(query: str) -> str:
        # Remove commas and parentheses that commonly break simple queries
        q = re.sub(r"[(),]", " ", query)
        q = re.sub(r"\s+", " ", q).strip()

        # Map common organism mentions to NCBI organism filter syntax
        if re.search(r"\b(human|homo sapiens)\b", q, re.I):
            q = f'{q} AND "Homo sapiens"[Organism]'
        elif re.search(r"\b(mouse|mus musculus|murine)\b", q, re.I):
            q = f'{q} AND "Mus musculus"[Organism]'
        elif re.search(r"\b(rat|rattus norvegicus)\b", q, re.I):
            q = f'{q} AND "Rattus norvegicus"[Organism]'

        # Normalize some common terms
        q = q.replace("single cell", "single-cell")
        q = q.replace("rna seq", "RNA-seq")

        return q

    if not openai_key:
        return _heuristic(user_query)

    # Try calling OpenAI Responses API (best-effort). If anything fails,
    # fall back to the local heuristic.
    try:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        body = {"model": "gpt-5", "input": prompt}
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Attempt to extract text robustly from Responses API schema
        term = ""
        if isinstance(data.get("output_text"), str):
            term = data.get("output_text", "").strip()
        else:
            outputs = data.get("output", [])
            texts = []
            for out in outputs:
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
            term = " ".join(texts).strip()

        if not term:
            # Last-resort: try 'text' fields nested differently
            if isinstance(data.get("choices"), list) and data["choices"]:
                term = data["choices"][0].get("text", "").strip()

        if term:
            return term
    except Exception:
        # ignore and fall back
        pass

    return _heuristic(user_query)


# ---------- 2. Low-level GEO / Entrez helpers ----------

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI asks you to identify yourself and (optionally) use an API key
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "saleemrji@gmail.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # optional but recommended


def _add_ncbi_params(params: dict) -> dict:
    params = dict(params)  # copy
    params.setdefault("email", NCBI_EMAIL)
    if NCBI_API_KEY:
        params.setdefault("api_key", NCBI_API_KEY)
    return params


def _extract_source_type(rec: dict) -> str:
    """
    Extract tissue, strain, cell line, or any other biological source information from NCBI record.
    Looks in samples titles, sample information, and summary text.
    """
    source_types = set()

    # Comprehensive list of source type keywords
    tissue_keywords = [
        "tissue",
        "brain",
        "tumor",
        "blood",
        "bone",
        "muscle",
        "heart",
        "liver",
        "kidney",
        "lung",
        "skin",
        "eye",
        "ear",
        "nerve",
        "gut",
        "intestine",
        "stomach",
        "pancreas",
        "spleen",
        "thymus",
        "lymph",
        "thyroid",
        "adrenal",
        "hypothalamus",
        "pituitary",
        "cerebellum",
        "cortex",
        "hippocampus",
    ]

    organism_keywords = ["strain", "line", "cultivar", "variety", "ecotype"]

    sample_source_keywords = [
        "primary",
        "cultured",
        "established",
        "cell culture",
        "organoid",
        "spheroid",
        "explant",
        "biopsy",
        "autopsy",
        "derived",
        "dissociated",
    ]

    # Check samples for any source-related info
    samples = rec.get("samples", [])
    if samples:
        for sample in samples:
            title = sample.get("title", "").lower()

            # Check for tissue types
            for keyword in tissue_keywords:
                if keyword in title:
                    source_types.add(keyword)

            # Check for organism/strain types
            for keyword in organism_keywords:
                if keyword in title:
                    source_types.add(keyword)

            # Check for culture/preparation types
            for keyword in sample_source_keywords:
                if keyword in title:
                    source_types.add(keyword)

    # If found in samples, return them
    if source_types:
        return "; ".join(sorted(source_types))

    # Fallback: check title
    title = rec.get("title", "").lower()
    for keyword in tissue_keywords + organism_keywords + sample_source_keywords:
        if keyword in title:
            source_types.add(keyword)

    if source_types:
        return "; ".join(sorted(source_types))

    # Final fallback: check summary for keywords
    summary = rec.get("summary", "").lower()
    for keyword in tissue_keywords + organism_keywords + sample_source_keywords:
        if keyword in summary:
            source_types.add(keyword)

    return "; ".join(sorted(source_types)) if source_types else "not specified"


def esearch_gds(term: str, max_results: int = 50) -> list[str]:
    """
    Call eSearch on db=gds, restricted to Series records (gse[ETYP]),
    and return the list of internal UIDs.
    """
    # Restrict to Series records with gse[ETYP] as per NCBI examples. :contentReference[oaicite:2]{index=2}
    full_term = f"({term}) AND gse[ETYP]"
    params = _add_ncbi_params(
        {
            "db": "gds",
            "term": full_term,
            "retmode": "json",
            "retmax": max_results,
        }
    )
    r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["esearchresult"]
    return data.get("idlist", [])


def esummary_gds(uids: list[str]) -> pd.DataFrame:
    """
    Call eSummary on db=gds UIDs and return a tidy DataFrame with GSE accessions.
    """
    if not uids:
        return pd.DataFrame(
            columns=[
                "uid",
                "gse_accession",
                "title",
                "summary",
                "taxon",
                "gds_type",
                "source_type",
                "n_samples",
                "pubmed_ids",
                "ftp_link",
            ]
        )

    params = _add_ncbi_params(
        {
            "db": "gds",
            "id": ",".join(uids),
            "retmode": "json",
            "version": "2.0",
        }
    )
    r = requests.get(f"{NCBI_BASE}/esummary.fcgi", params=params, timeout=30)
    r.raise_for_status()
    result = r.json()["result"]

    rows = []
    for uid in result["uids"]:
        rec = result[uid]
        # Field names like 'accession', 'title', 'taxon', 'gdstype',
        # 'n_samples', 'pubmedids', 'ftplink' come from the JSON schema. :contentReference[oaicite:3]{index=3}
        rows.append(
            {
                "uid": uid,
                "gse_accession": rec.get("accession"),  # e.g. "GSE256370"
                "title": rec.get("title"),
                "summary": rec.get("summary"),
                "taxon": rec.get("taxon"),
                "gds_type": rec.get(
                    "gdstype"
                ),  # e.g. "Expression profiling by high throughput sequencing"
                "source_type": _extract_source_type(
                    rec
                ),  # e.g. "tissue; tumor" or "cell line"
                "n_samples": rec.get("n_samples"),
                "pubmed_ids": ",".join(rec.get("pubmedids", [])),
                "ftp_link": rec.get("ftplink"),
            }
        )

    return pd.DataFrame(rows)


# ---------- 3. High-level GenAI + GEO function ----------


def search_gse_with_genai(user_query: str, max_results: int = 50) -> pd.DataFrame:
    """
    Full pipeline:
      1) use GenAI to build an esearch term
      2) run esearch on GEO DataSets (db=gds, Series only)
      3) run esummary to get GSE IDs + metadata
    """
    esearch_term = expand_query_with_llm(user_query)
    uids = esearch_gds(esearch_term, max_results=max_results)
    df = esummary_gds(uids)

    # Keep only proper GSE entries
    df = df[df["gse_accession"].str.startswith("GSE", na=False)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    query = "single-cell RNA-seq pediatric medulloblastoma"
    df = search_gse_with_genai(query, max_results=30)
    df.to_csv("gse_serch_results.csv", index=False)
    print(
        df[["gse_accession", "title", "source_type", "taxon", "gds_type", "n_samples"]]
    )
