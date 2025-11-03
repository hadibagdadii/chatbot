from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from collections import Counter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from .config import (OLLAMA_MODEL, OLLAMA_HOST, VSTORE_DIR, TOP_N_DOCS, DATE_COL, TOP_K)

def _combined_text(row: pd.Series) -> str:
    return " | ".join([
        f"Part: {row.get('part_number', '')}",
        f"Type: {row.get('typename', '')}",
        f"Station: {row.get('stationnumber', '')} {row.get('stationdescription','')}",
        f"Failure Code: {row.get('failure_code', '')}",
        f"Failure Description: {row.get('failure_description', '')}",
        f"Defect: {row.get('defect', '')}",
        f"Failure Details: {row.get('failure_details', '')}",
        f"Action Code: {row.get('action_code','')}",
        f"Material: {row.get('material_code','')} {row.get('material_desc','')}",
        f"PartClass: {row.get('partclass','')}",
        f"Serial: {row.get('serialnumber','')}",
        f"Date: {row.get('date','')}",
    ])

def build_documents(df: pd.DataFrame) -> List[Document]:
    docs = []
    for _, r in df.iterrows():
        txt = _combined_text(r)
        meta = {k: (r.get(k) if k in r else None) for k in r.index}
        docs.append(Document(page_content=txt, metadata=meta))
    return docs

def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)

def create_or_load_faiss(df: pd.DataFrame, vstore_dir: Path = VSTORE_DIR) -> FAISS:
    vstore_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder()
    try:
        return FAISS.load_local(vstore_dir.as_posix(), embedder, allow_dangerous_deserialization=True)
    except Exception:
        docs = build_documents(df)
        vs = FAISS.from_documents(docs, embedder)
        vs.save_local(vstore_dir.as_posix())
        return vs

def retrieve_semantic(query: str, vectorstore, top_n: int = TOP_N_DOCS) -> Dict[str, Any]:
    """
    Retrieve relevant documents and aggregate insights.
    
    Returns aggregated Action Codes, Materials, and recurring serials.
    """
    # Semantic search
    results = vectorstore.similarity_search(query, k=top_n)
    
    if not results:
        return {
            "message": "No relevant historical data found.",
            "action_codes": [],
            "materials": [],
            "recurring_serials": []
        }
    
    # Extract metadata
    action_codes = []
    materials = []
    serials = []
    dates = []
    
    for doc in results:
        meta = doc.metadata
        
        # Collect action codes
        if meta.get("action_code"):
            action_codes.append(str(meta["action_code"]))
        
        # Collect materials
        mat_code = meta.get("material_code", "")
        mat_desc = meta.get("material_desc", "")
        part_class = meta.get("partclass", "")
        if mat_code or mat_desc:
            materials.append(f"{mat_code} - {mat_desc} ({part_class})".strip(" -()"))
        
        # Collect serials and dates for recency analysis
        if meta.get("serialnumber"):
            serials.append(str(meta["serialnumber"]))
        if meta.get(DATE_COL):
            dates.append(str(meta[DATE_COL]))
    
    # Aggregate with counts
    action_counter = Counter(action_codes)
    material_counter = Counter(materials)
    serial_counter = Counter(serials)
    
    # Format results
    top_actions = [
        f"{code} (seen {count}x)" 
        for code, count in action_counter.most_common(TOP_K)
    ]
    
    top_materials = [
        f"{mat} (used {count}x)" 
        for mat, count in material_counter.most_common(TOP_K)
    ]
    
    recurring = [
        f"{serial} (appears {count}x)" 
        for serial, count in serial_counter.most_common(TOP_K) 
        if count > 1
    ]
    
    return {
        "retrieved_count": len(results),
        "action_codes": top_actions or ["No action codes found"],
        "materials": top_materials or ["No materials recorded"],
        "recurring_serials": recurring or ["No recurring issues detected"],
        "date_range": f"{min(dates) if dates else 'N/A'} to {max(dates) if dates else 'N/A'}"
    }