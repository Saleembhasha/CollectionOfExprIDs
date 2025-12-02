import os
import requests
import pandas as pd
import re

# ---------- 1. GenAI helper ----------


def expand_query_with_llm(user_query: str) -> str:
    """
    Take a natural-language query and turn it into a GEO DataSets esearch term.

    Replace the body with a call to your GenAI provider (OpenAI, etc.).
    For now, we just return the raw query as a fallback.
    """
    # Example (OpenAI Responses API), commented so it doesn't break if you
    # haven't configured it yet:
    #
    # from openai import OpenAI
    # client = OpenAI()
    # prompt = f"""
    # You build NCBI GEO DataSets (db=gds) esearch terms.
    # Convert this user query into a single esearch term that:
    #   - will be used with db=gds
    #   - restricts to Series (use gse[ETYP])
    #   - includes organism / data type if present
    # Return only the term, nothing else.
    #
    # Query: {user_query!r}
    # """
    # resp = client.responses.create(
    #     model="gpt-5",
    #     input=prompt,
    # )
    # term = resp.output_text.strip()
    # return term
    return user_query  # no-op fallback if you haven't wired GenAI yet


# ---------- 2. Low-level GEO / Entrez helpers ----------

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI asks you to identify yourself and (optionally) use an API key. :contentReference[oaicite:1]{index=1}
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "saleemrji@gmail.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")  # optional but recommended


def _add_ncbi_params(params: dict) -> dict:
    params = dict(params)  # copy
    params.setdefault("email", NCBI_EMAIL)
    if NCBI_API_KEY:
        params.setdefault("api_key", NCBI_API_KEY)
    return params


def esearch_gds(
    term: str,
    max_results: int = 50,
    organism: str = None,
    tissue: str = None,
    cell_line: str = None,
) -> list[str]:
    """
    Call eSearch on db=gds, restricted to Series records (gse[ETYP]),
    and return the list of internal UIDs.

    Optional attribute filters:
        organism: Organism filter using [ORGN] attribute (e.g., "Homo sapiens")
        tissue: Tissue type filter using [TISSUE] attribute (e.g., "brain", "liver")
        cell_line: Cell line filter using [CELL_LINE] attribute (e.g., "HeLa", "HEK293")
    """
    # Restrict to Series records with gse[ETYP] as per NCBI examples
    full_term = f"({term}) AND gse[ETYP]"

    # Add organism filter if provided
    if organism:
        full_term = f"{full_term} AND {organism}[ORGN]"

    # Add tissue filter if provided
    if tissue:
        full_term = f"{full_term} AND {tissue}[TISSUE]"

    # Add cell line filter if provided
    if cell_line:
        full_term = f"{full_term} AND {cell_line}[CELL_LINE]"

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
    Includes a single 'sample_source' column that consolidates tissue, strain, and cell_line info.
    """
    if not uids:
        return pd.DataFrame(
            columns=[
                "uid",
                "gse_accession",
                "title",
                "summary",
                "taxon",
                "organism_attr",
                "sample_source",
                "gds_type",
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
        # Extract organism from taxon field
        organism_str = rec.get("taxon", "Not specified")

        # Extract sample source from title and summary
        # Look for keywords indicating tissue, cell line, or strain
        title = rec.get("title", "").lower()
        summary = rec.get("summary", "").lower()
        # Also include overall design / design fields if available
        overall_design = rec.get("overall_design", "") or rec.get("design", "")
        overall_design = (
            overall_design.lower() if isinstance(overall_design, str) else ""
        )
        full_text = " ".join([title, summary, overall_design])

        sample_sources = []

        # Common tissue keywords
        tissue_keywords = {
            "brain": "brain",
            "liver": "liver",
            "heart": "heart",
            "kidney": "kidney",
            "lung": "lung",
            "blood": "blood",
            "skin": "skin",
            "muscle": "muscle",
            "pancreas": "pancreas",
            "breast": "breast",
            "prostate": "prostate",
            "colon": "colon",
            "stomach": "stomach",
            "adipose": "adipose tissue",
            "bone marrow": "bone marrow",
            "spleen": "spleen",
            "thymus": "thymus",
            "lymph": "lymph node",
            "testis": "testis",
            "ovary": "ovary",
        }

        # Extract cell lines from text (title/summary/overall_design) using regex
        # Approach:
        #  - Look for phrases followed by 'cell(s)' or 'cell line' (e.g. 'HeLa cells', 'D283 Med cells')
        #  - Also capture common uppercase alphanumeric patterns (HEK293, A549, K562, MCF7, 293T)
        #  - Use a set to deduplicate and normalize tokens
        original_text = " ".join(
            [rec.get("title", ""), rec.get("summary", ""), overall_design]
        )
        token_candidates = set()

        # Context matches: look before 'cells' / 'cell line' and split into tokens
        for m in re.findall(
            r"([A-Za-z0-9\-_/]{2,80}?)\s+(?:cell line|cells|cell|cell-line)\b",
            original_text,
            flags=re.I,
        ):
            # choose concise candidate(s) from the phrase immediately before 'cells'
            toks = re.findall(r"[A-Za-z0-9\-]+", m)
            if not toks:
                continue
            candidate = None
            last = toks[-1]
            if re.search(r"\d", last) or (
                re.search(r"[A-Z]", last)
                and re.search(r"[a-z]", last)
                and 3 <= len(last) <= 10
            ):
                candidate = last
            elif len(toks) >= 2 and re.search(r"\d", toks[-2]) and len(last) <= 8:
                candidate = toks[-2] + " " + last
            else:
                for tok in reversed(toks):
                    if re.search(r"\d", tok):
                        candidate = tok
                        break
            if candidate:
                token_candidates.add(candidate)

        # Uppercase alphanumeric patterns (HEK293, A549, K562, MCF7, 293T)
        for m in re.findall(
            r"\b(?:[A-Z]{2,}[0-9]+[A-Z0-9\-]*|[0-9]{2,}T)\b", original_text
        ):
            token_candidates.add(m)

        # Normalize and filter tokens: require digit or mixed-case short name
        normalized_cell_lines = []
        for tok in sorted(token_candidates, key=lambda x: x.lower()):
            t = tok.strip().strip(" ,;:.\n\t")
            if len(t) > 25:
                continue
            if re.search(r"\d", t) or (
                re.search(r"[A-Z]", t) and re.search(r"[a-z]", t) and 3 <= len(t) <= 10
            ):
                normalized_cell_lines.append(t)

        normalized_cell_lines = list(dict.fromkeys(normalized_cell_lines))

        # Common strain keywords (for animal studies)
        strain_keywords = [
            "c57bl",
            "balb/c",
            "fvb",
            "129",
            "knockout",
            "transgenic",
            "wt",
        ]

        # Check for tissues
        for keyword, tissue_type in tissue_keywords.items():
            if keyword in full_text:
                sample_sources.append(f"Tissue: {tissue_type.title()}")
                break

        # Add detected cell lines (from summary/overall_design/title)
        if normalized_cell_lines:
            sample_sources.append(f"Cell Line: {', '.join(normalized_cell_lines)}")

        # Check for strains
        for strain in strain_keywords:
            if strain in full_text:
                sample_sources.append(f"Strain: {strain.title()}")
                break

        sample_source_str = (
            " | ".join(sample_sources) if sample_sources else "Not specified"
        )

        rows.append(
            {
                "uid": uid,
                "gse_accession": rec.get("accession"),  # e.g. "GSE256370"
                "title": rec.get("title"),
                "summary": rec.get("summary"),
                "taxon": rec.get("taxon"),
                "organism_attr": organism_str,
                "sample_source": sample_source_str,
                "gds_type": rec.get(
                    "gdstype"
                ),  # e.g. "Expression profiling by high throughput sequencing"
                "n_samples": rec.get("n_samples"),
                "pubmed_ids": ",".join(rec.get("pubmedids", [])),
                "ftp_link": rec.get("ftplink"),
            }
        )

    return pd.DataFrame(rows)


# ---------- 3. EMBL-Array Expression Web helpers ----------

EMBL_ARRAY_BASE = "https://www.ebi.ac.uk/arrayexpress"


def search_embl_array_expression(
    query: str, organism: str = "Homo sapiens", max_results: int = 50
) -> pd.DataFrame:
    """
    Search EMBL-Array Expression Web for microarray and RNA-seq experiments.
    Returns a DataFrame with experiment details.
    """
    # EMBL ArrayExpress REST API
    # https://www.ebi.ac.uk/arrayexpress/help/query-rest-api.html

    try:
        # Try simple query first without complex filtering
        params = {
            "query": query,
            "pagesize": max_results,
            "sortby": "releasedate",
            "format": "json",
        }

        url = f"{EMBL_ARRAY_BASE}/json/experiments"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        rows = []
        experiments = data.get("experiments", {}).get("experiment", [])

        # Handle single experiment response (wrap in list)
        if isinstance(experiments, dict):
            experiments = [experiments]

        for exp in experiments:
            # Extract relevant fields
            rows.append(
                {
                    "accession": exp.get("accession"),
                    "title": exp.get("name"),
                    "description": exp.get("description", ""),
                    "organism": exp.get("organism"),
                    "exp_type": exp.get("experimenttype"),
                    "releasedate": exp.get("releasedate"),
                    "lastupdate": exp.get("lastupdate"),
                    "assays": exp.get("assaycount"),
                    "source": "EMBL-Array",
                }
            )

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Warning: EMBL-Array search unavailable ({str(e)[:50]}...)")
        return pd.DataFrame(
            columns=[
                "accession",
                "title",
                "description",
                "organism",
                "exp_type",
                "releasedate",
                "lastupdate",
                "assays",
                "source",
            ]
        )


def filter_human_embl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter EMBL-Array results to keep only human experiments.
    """
    if df.empty:
        return df
    human_df = df[
        df["organism"].str.contains("Homo sapiens", case=False, na=False)
    ].reset_index(drop=True)
    return human_df


# ---------- 4. High-level GenAI + GEO function ----------


def search_gse_with_genai(
    user_query: str,
    max_results: int = 50,
    organism: str = None,
    tissue: str = None,
    cell_line: str = None,
    restrict_human: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline:
      1) use GenAI to build an esearch term
      2) run esearch on GEO DataSets (db=gds, Series only)
      3) run esummary to get GSE IDs + metadata

    Args:
        user_query: Search term (disease, gene, etc.)
        max_results: Maximum number of results
        organism: Organism attribute filter (e.g., "Homo sapiens", "Mus musculus")
        tissue: Tissue type attribute filter (e.g., "brain", "liver", "kidney")
        cell_line: Cell line attribute filter (e.g., "HeLa", "HEK293")
        restrict_human: If True, automatically set organism to "Homo sapiens"
    """
    esearch_term = expand_query_with_llm(user_query)

    # Use organism attribute in search if provided or restrict_human is True
    organism_filter = "Homo sapiens" if restrict_human else organism
    uids = esearch_gds(
        esearch_term,
        max_results=max_results,
        organism=organism_filter,
        tissue=tissue,
        cell_line=cell_line,
    )
    df = esummary_gds(uids)

    # Keep only proper GSE entries
    df = df[df["gse_accession"].str.startswith("GSE", na=False)].reset_index(drop=True)
    return df


def filter_human_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to keep only human (Homo sapiens) records.
    """
    human_df = df[
        df["taxon"].str.contains("Homo sapiens", case=False, na=False)
    ].reset_index(drop=True)
    return human_df


def filter_partial_match(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter DataFrame to keep only records that partially match the query
    in title, summary, or taxon fields.
    """
    query_lower = query.lower()
    mask = (
        df["title"].str.contains(query_lower, case=False, na=False)
        | df["summary"].str.contains(query_lower, case=False, na=False)
        | df["taxon"].str.contains(query_lower, case=False, na=False)
    )
    matched_df = df[mask].reset_index(drop=True)
    return matched_df


def search_with_attributes(
    user_query: str,
    max_results: int = 50,
    organism: str = "Homo sapiens",
    tissue: str = None,
    cell_line: str = None,
) -> pd.DataFrame:
    """
    Search GEO with specific attribute filters for more targeted results.

    Supported GEO Attributes:
      - organism: [ORGN] - "Homo sapiens", "Mus musculus", "Drosophila melanogaster", etc.
      - tissue: [TISSUE] - "brain", "liver", "heart", "kidney", "blood", "skin", etc.
      - cell_line: [CELL_LINE] - "HeLa", "HEK293", "K562", "MCF7", etc.

    Example:
        search_with_attributes("cancer", organism="Homo sapiens", tissue="brain")
        search_with_attributes("differentiation", cell_line="HEK293")
    """
    df = search_gse_with_genai(
        user_query,
        max_results=max_results,
        organism=organism,
        tissue=tissue,
        cell_line=cell_line,
    )
    return df


def search_specific_experiment(experiment_id: str) -> pd.DataFrame:
    """
    Search for a specific experiment by GSE ID or other identifier.
    Returns a DataFrame with the matching experiment details.
    """
    # Try to fetch the experiment directly if it's a GSE ID
    q = experiment_id.strip()
    if q.upper().startswith("GSE"):
        try:
            uids = esearch_gds(q, max_results=5)
            if uids:
                df = esummary_gds(uids)
                return df[df["gse_accession"].str.upper() == q.upper()].reset_index(
                    drop=True
                )
        except Exception:
            pass

    # If not a GSE ID, try to detect a gds-type phrase in the query
    q_lower = q.lower()
    gds_type_keywords = [
        "expression profiling by high throughput sequencing",
        "expression profiling by array",
        "expression profiling by high throughput",
        "expression profiling",
        "rna-seq",
        "microarray",
        "methylation profiling",
        "single-cell",
        "single cell",
    ]

    detected_type = None
    for k in gds_type_keywords:
        if k in q_lower:
            detected_type = k
            # remove the phrase from the main query to make search broader
            q = q_lower.replace(k, " ")
            break

    # Run a normal attribute-aware search (loose) using the remaining query
    df = search_gse_with_genai(q.strip(), max_results=200)

    # If a gds type was detected, filter the results by that gds_type column
    if detected_type and not df.empty:
        mask = df["gds_type"].str.contains(
            detected_type.split()[0], case=False, na=False
        )
        filtered = df[mask].reset_index(drop=True)
        if not filtered.empty:
            return filtered

    # Fallbacks: try exact title match (case-insensitive substring)
    if not df.empty:
        mask_title = df["title"].str.contains(q.strip(), case=False, na=False)
        if mask_title.any():
            return df[mask_title].reset_index(drop=True)

    # Last fallback: partial match across title/summary/taxon
    df_partial = filter_partial_match(df, experiment_id)
    return df_partial


if __name__ == "__main__":
    import sys

    # Check if a specific experiment was requested via command line
    if len(sys.argv) > 1:
        experiment_query = " ".join(sys.argv[1:])
        print(f"\nSearching for specific experiment: {experiment_query}")
        specific_df = search_specific_experiment(experiment_query)

        if len(specific_df) > 0:
            print(f"\nFound {len(specific_df)} result(s):")
            print(
                specific_df[
                    [
                        "gse_accession",
                        "title",
                        "taxon",
                        "gds_type",
                        "n_samples",
                        "summary",
                    ]
                ].to_string()
            )
            specific_df.to_csv(
                f"geo_specific_experiment_{experiment_query.replace(' ', '_')}.csv",
                index=False,
            )
            print(
                f"\nResults saved to: geo_specific_experiment_{experiment_query.replace(' ', '_')}.csv"
            )
        else:
            print(f"No experiments found matching: {experiment_query}")
    else:
        # Default behavior: Search for medulloblastoma records in both GEO and EMBL
        query = "medulloblastoma"
        print(f"\n{'=' * 70}")
        print(f"Searching for: {query}")
        print(f"{'=' * 70}")

        # GEO search with organism attribute filter
        print(f"\n[1] Searching GEO DataSets (using organism attribute)...")
        # Search directly with organism attribute to get human records
        geo_df = search_gse_with_genai(query, max_results=100, restrict_human=True)
        print(f"Total GEO records found (Homo sapiens): {len(geo_df)}")

        # Additional filtering for partial match with query term
        geo_partial_match_df = filter_partial_match(geo_df, query)
        print(
            f"GEO records with partial match ('{query}'): {len(geo_partial_match_df)}"
        )

        # Save GEO results
        geo_df.to_csv("geo_embl_ids_human.csv", index=False)
        geo_partial_match_df.to_csv("geo_embl_ids_human_partial_match.csv", index=False)

        # Search with tissue attribute filter (brain tissue)
        print(f"\n[1b] Searching GEO with tissue attribute (brain)...")
        geo_brain_df = search_with_attributes(
            query, max_results=50, organism="Homo sapiens", tissue="brain"
        )
        print(f"GEO records found (Homo sapiens + brain tissue): {len(geo_brain_df)}")
        if len(geo_brain_df) > 0:
            geo_brain_df.to_csv("geo_embl_ids_human_brain.csv", index=False)

        # EMBL-Array search
        print(f"\n[2] Searching EMBL-Array Expression...")
        embl_df = search_embl_array_expression(
            query, organism="Homo sapiens", max_results=100
        )
        print(f"Total EMBL-Array records found: {len(embl_df)}")

        if len(embl_df) > 0:
            embl_human_df = filter_human_embl(embl_df)
            print(f"Human EMBL-Array records: {len(embl_human_df)}")
            embl_human_df.to_csv("embl_array_human.csv", index=False)
        else:
            embl_human_df = pd.DataFrame()
            print("No EMBL-Array records found")

        # Combined results
        print(f"\n{'=' * 70}")
        print(f"SUMMARY:")
        print(f"{'=' * 70}")
        print(f"GEO (Homo sapiens - organism attribute): {len(geo_df)} records")
        print(f"GEO (Homo sapiens + brain tissue): {len(geo_brain_df)} records")
        print(f"EMBL-Array (human): {len(embl_human_df)} records")
        print(f"Total: {len(geo_df) + len(geo_brain_df) + len(embl_human_df)} records")

        if len(geo_df) > 0:
            print(f"\nTop GEO human results:")
            print(
                geo_df[
                    [
                        "gse_accession",
                        "title",
                        "organism_attr",
                        "sample_source",
                        "gds_type",
                        "n_samples",
                    ]
                ].head(10)
            )

        if len(geo_brain_df) > 0:
            print(f"\nTop GEO brain tissue results (Homo sapiens):")
            print(
                geo_brain_df[
                    [
                        "gse_accession",
                        "title",
                        "organism_attr",
                        "sample_source",
                        "gds_type",
                        "n_samples",
                    ]
                ].head(5)
            )

        if len(embl_human_df) > 0:
            print(f"\nTop EMBL-Array human results:")
            print(
                embl_human_df[
                    ["accession", "title", "organism", "exp_type", "assays"]
                ].head(10)
            )

        print(f"\n--- Files saved ---")
        print(f"geo_embl_ids_human.csv (Homo sapiens)")
        print(f"geo_embl_ids_human_partial_match.csv (Homo sapiens + partial match)")
        if len(geo_brain_df) > 0:
            print(f"geo_embl_ids_human_brain.csv (Homo sapiens + brain tissue)")
        if len(embl_human_df) > 0:
            print(f"embl_array_human.csv (EMBL-Array Homo sapiens)")

        print(f"\n--- Supported GEO Attributes ---")
        print(
            f"  organism: [ORGN] - 'Homo sapiens', 'Mus musculus', 'Drosophila melanogaster', etc."
        )
        print(
            f"  tissue: [TISSUE] - 'brain', 'liver', 'heart', 'kidney', 'blood', 'skin', etc."
        )
        print(f"  cell_line: [CELL_LINE] - 'HeLa', 'HEK293', 'K562', 'MCF7', etc.")

        print(f"\n--- Usage examples with attributes ---")
        print(f"# In Python:")
        print(f"from geo_enmbl_ids_extraction import search_with_attributes")
        print(
            f"df = search_with_attributes('cancer', organism='Homo sapiens', tissue='brain')"
        )
        print(f"df = search_with_attributes('differentiation', cell_line='HEK293')")
