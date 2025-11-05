from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from collections import Counter
import logging
import re
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from .config import (OLLAMA_MODEL, OLLAMA_HOST, VSTORE_DIR, TOP_N_DOCS, DATE_COL, TOP_K)

logger = logging.getLogger(__name__)

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
    
    # Try to load existing vectorstore
    try:
        logger.info("Attempting to load existing vectorstore...")
        vs = FAISS.load_local(vstore_dir.as_posix(), embedder, allow_dangerous_deserialization=True)
        logger.info(f"✓ Loaded existing vectorstore with {vs.index.ntotal} vectors")
        return vs
    except Exception as e:
        logger.info(f"No existing vectorstore found, creating new one...")
        logger.info(f"This will take several minutes for {len(df)} records...")
        
        # Build documents
        docs = build_documents(df)
        logger.info(f"Built {len(docs)} documents, starting embedding process...")
        
        # Create vectorstore with progress indication
        try:
            # Process in smaller batches to show progress
            batch_size = 100
            vectorstore = None
            
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(docs) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
                
                if vectorstore is None:
                    # Create initial vectorstore
                    vectorstore = FAISS.from_documents(batch, embedder)
                else:
                    # Add to existing vectorstore
                    new_vs = FAISS.from_documents(batch, embedder)
                    vectorstore.merge_from(new_vs)
                
                logger.info(f"✓ Batch {batch_num}/{total_batches} complete")
            
            # Save the vectorstore
            logger.info("Saving vectorstore to disk...")
            vectorstore.save_local(vstore_dir.as_posix())
            logger.info(f"✓ Vectorstore saved with {vectorstore.index.ntotal} vectors")
            
            return vectorstore
            
        except Exception as embed_error:
            logger.error(f"Failed to create embeddings: {embed_error}")
            logger.error("Is Ollama running? Try: ollama serve")
            raise

def _clean_value(val) -> str:
    """Clean up nan/None values for display"""
    if pd.isna(val) or val is None or str(val).lower() == 'nan':
        return ""
    return str(val).strip()

def _extract_part_numbers(query: str) -> List[str]:
    """Extract part numbers from query (e.g., 10003939)"""
    # Look for patterns like 10003939, 1000679-001, etc.
    patterns = re.findall(r'\b\d{7,10}(?:-\d{3})?\b', query)
    return patterns

def retrieve_semantic(query: str, vectorstore, top_n: int = TOP_N_DOCS) -> Dict[str, Any]:
    """
    Retrieve relevant documents and aggregate insights with better filtering and accuracy.
    """
    logger.info(f"Running semantic search for: {query}")
    
    # Extract any specific part numbers mentioned
    mentioned_parts = _extract_part_numbers(query)
    logger.info(f"Detected part numbers in query: {mentioned_parts}")
    
    # If part numbers are mentioned, get way more results for better coverage
    search_k = top_n * 3 if mentioned_parts else top_n * 2
    
    # Semantic search
    results = vectorstore.similarity_search(query, k=search_k)
    
    if not results:
        return {
            "message": "No relevant historical data found.",
            "query_context": query,
            "action_codes": [],
            "materials": [],
            "recurring_serials": [],
            "failure_codes": []
        }
    
    # Filter results by part number if specified
    if mentioned_parts:
        filtered_results = []
        for doc in results:
            doc_part = _clean_value(doc.metadata.get('part_number', ''))
            if any(part in doc_part for part in mentioned_parts):
                filtered_results.append(doc)
        
        if filtered_results:
            results = filtered_results
            logger.info(f"Filtered to {len(results)} results matching part numbers: {mentioned_parts}")
        else:
            logger.warning(f"No results found for part numbers {mentioned_parts}, using general search")
            results = results[:top_n]
    else:
        results = results[:top_n]
    
    # Extract metadata with better handling
    action_codes = []
    materials_list = []
    material_details = {}  # Track full material info
    serials = []
    dates = []
    failure_codes = []
    stations = []
    
    for doc in results:
        meta = doc.metadata
        
        # Collect action codes (skip nan)
        action_code = _clean_value(meta.get("action_code"))
        if action_code:
            action_codes.append(action_code)
        
        # Collect failure codes
        failure_code = _clean_value(meta.get("failure_code"))
        if failure_code:
            failure_codes.append(failure_code)
        
        # Collect stations
        station = _clean_value(meta.get("stationnumber"))
        if station:
            stations.append(station)
        
        # Collect materials with full details
        mat_code = _clean_value(meta.get("material_code"))
        mat_desc = _clean_value(meta.get("material_desc"))
        part_class = _clean_value(meta.get("partclass"))
        
        if mat_code:  # Only include if there's an actual material code
            materials_list.append(mat_code)
            # Store full details for later
            if mat_code not in material_details:
                material_details[mat_code] = {
                    'code': mat_code,
                    'description': mat_desc if mat_desc else 'N/A',
                    'part_class': part_class if part_class else 'N/A',
                    'count': 0
                }
            material_details[mat_code]['count'] += 1
        
        # Collect serials
        serial = _clean_value(meta.get("serialnumber"))
        if serial:
            serials.append(serial)
        
        # Collect dates
        date = _clean_value(meta.get(DATE_COL))
        if date:
            dates.append(date)
    
    # Aggregate with counts
    action_counter = Counter(action_codes)
    failure_counter = Counter(failure_codes)
    serial_counter = Counter(serials)
    station_counter = Counter(stations)
    
    # Format results - Top action codes
    top_actions = [
        {"code": code, "count": count}
        for code, count in action_counter.most_common(TOP_K)
    ]
    
    # Format results - Top failure codes
    top_failures = [
        {"code": code, "count": count}
        for code, count in failure_counter.most_common(TOP_K)
    ]
    
    # Format results - Top materials (sorted by count)
    sorted_materials = sorted(
        material_details.values(),
        key=lambda x: x['count'],
        reverse=True
    )[:TOP_K]
    
    # Format results - Recurring serials (only those that appear 2+ times)
    recurring = [
        {"serial": serial, "count": count}
        for serial, count in serial_counter.most_common(10)
        if count > 1
    ]
    
    # Most common stations
    top_stations = [
        {"station": station, "count": count}
        for station, count in station_counter.most_common(3)
    ]
    
    logger.info(f"Retrieved {len(results)} documents, found {len(top_actions)} action codes, {len(sorted_materials)} materials, {len(recurring)} recurring serials")
    
    return {
        "retrieved_count": len(results),
        "query_context": query,
        "mentioned_parts": mentioned_parts,
        "action_codes": top_actions if top_actions else [{"note": "No action codes found"}],
        "failure_codes": top_failures if top_failures else [{"note": "No failure codes found"}],
        "materials": sorted_materials if sorted_materials else [{"note": "No materials recorded"}],
        "recurring_serials": recurring if recurring else [{"note": "No recurring serials detected"}],
        "common_stations": top_stations,
        "date_range": {
            "earliest": min(dates) if dates else "N/A",
            "latest": max(dates) if dates else "N/A"
        }
    }