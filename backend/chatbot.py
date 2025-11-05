from typing import Dict, Any, Generator
from langchain_community.llms import Ollama
from .config import OLLAMA_MODEL, OLLAMA_HOST
import re

# Keywords that indicate a real failure/technical query
TECHNICAL_KEYWORDS = [
    'failure', 'fail', 'error', 'defect', 'issue', 'problem', 'broken',
    'leak', 'test', 'station', 'code', 'serial', 'part', 'material',
    'action', 'fix', 'repair', 'replace', 'malfunction', 'fault',
    'overheating', 'not working', 'doesn\'t work', 'stopped',
    'diagnostic', 'troubleshoot', 'debug', 'ncr', 'duplicate',
    'recurring', 'reoccurring', 'common', 'frequent', 'match', 'using'
]

SYSTEM_PROMPT_TECHNICAL = """You are a precise manufacturing support assistant analyzing historical failure data.

CRITICAL RULES:
1. ONLY use the data provided - never make up part numbers, codes, or materials
2. Be direct and conversational - no formal headers like "Analysis Report" or "Summary of Findings"
3. Always cite counts (how many times each item appears)
4. If a material description shows "N/A", just state the material code
5. Give practical, actionable advice based on the data

Respond in a helpful, direct tone. Start with what you found, then give recommendations."""

SYSTEM_PROMPT_CASUAL = """You are a helpful manufacturing support assistant having a natural conversation.

Respond naturally without quotation marks. Be friendly and conversational.
Let users know you can help with manufacturing failures, defects, failure codes, and part issues.
Keep it brief and welcoming."""

def is_technical_query(query: str) -> bool:
    """
    Determine if a query is about technical/failure issues or just casual conversation.
    """
    query_lower = query.lower()
    
    # Check for technical keywords
    for keyword in TECHNICAL_KEYWORDS:
        if keyword in query_lower:
            return True
    
    # Check for code patterns (e.g., "1025", "failure code 1234")
    if re.search(r'\b\d{3,10}(?:-\d{3})?\b', query_lower):
        return True
    
    # Check for station references
    if re.search(r'station\s*\d+', query_lower, re.IGNORECASE):
        return True
    
    # If query is very short (< 3 words), likely casual
    if len(query.split()) < 3:
        return False
    
    return False

def format_context_for_llm(agg: Dict[str, Any]) -> str:
    """Format the aggregated data in a clear, structured way for the LLM"""
    
    parts = []
    
    # Add query context
    parts.append(f"USER QUERY: {agg.get('query_context', 'N/A')}")
    parts.append(f"RECORDS ANALYZED: {agg.get('retrieved_count', 0)}")
    
    if agg.get('mentioned_parts'):
        parts.append(f"SPECIFIC PARTS QUERIED: {', '.join(agg['mentioned_parts'])}")
    
    parts.append("\n---DATA FROM HISTORICAL LOGS---\n")
    
    # Action codes
    parts.append("MOST COMMON ACTION CODES:")
    action_codes = agg.get('action_codes', [])
    if action_codes and isinstance(action_codes[0], dict) and 'code' in action_codes[0]:
        for item in action_codes:
            parts.append(f"  • {item['code']} (seen {item['count']}x)")
    else:
        parts.append("  • No action codes found")
    
    # Failure codes
    failure_codes = agg.get('failure_codes', [])
    if failure_codes and isinstance(failure_codes[0], dict) and 'code' in failure_codes[0]:
        parts.append("\nMOST COMMON FAILURE CODES:")
        for item in failure_codes:
            parts.append(f"  • {item['code']} (seen {item['count']}x)")
    
    # Materials
    parts.append("\nMOST FREQUENTLY USED MATERIALS:")
    materials = agg.get('materials', [])
    if materials and isinstance(materials[0], dict) and 'code' in materials[0]:
        for mat in materials:
            code = mat['code']
            desc = mat['description']
            pclass = mat['part_class']
            count = mat['count']
            parts.append(f"  • Material Code: {code}")
            if desc != 'N/A':
                parts.append(f"    Description: {desc}")
            parts.append(f"    Part Class: {pclass}")
            parts.append(f"    Used {count}x in similar cases")
    else:
        parts.append("  • No materials recorded")
    
    # Recurring serials
    recurring = agg.get('recurring_serials', [])
    if recurring and isinstance(recurring[0], dict) and 'serial' in recurring[0]:
        parts.append("\nRECURRING SERIAL NUMBERS (potential repeat issues):")
        for item in recurring:
            parts.append(f"  • Serial {item['serial']} appeared {item['count']}x")
    else:
        parts.append("\nRECURRING SERIAL NUMBERS: None detected")
    
    # Stations
    stations = agg.get('common_stations', [])
    if stations and len(stations) > 0:
        parts.append("\nMOST COMMON STATIONS:")
        for item in stations:
            parts.append(f"  • Station {item['station']} ({item['count']}x)")
    
    # Date range
    date_range = agg.get('date_range', {})
    if date_range:
        parts.append(f"\nDATE RANGE: {date_range.get('earliest', 'N/A')} to {date_range.get('latest', 'N/A')}")
    
    return "\n".join(parts)

def stream_response(query: str, agg: Dict[str, Any] = None) -> Generator[str, None, None]:
    """
    Stream response based on query type.
    """
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    
    # Determine if this is a technical query
    is_technical = is_technical_query(query)
    
    if is_technical and agg and agg.get('retrieved_count', 0) > 0:
        # Technical query with context - full analysis
        context = format_context_for_llm(agg)
        prompt = f"""{SYSTEM_PROMPT_TECHNICAL}

{context}

Based on the data above, answer this question directly and conversationally: {query}

Give a helpful, practical response. Include:
- What the failure code means based on the data
- What action to take (cite the most common action code)
- What part to replace (cite the most commonly used material)
- Any patterns you notice (recurring serials, common stations)

Be direct and skip formal headers. Just answer the question naturally."""
        
    elif is_technical and (not agg or agg.get('retrieved_count', 0) == 0):
        # Technical query but no context found
        prompt = f"""You are a helpful manufacturing support assistant.

The user asked: {query}

No matching historical data was found. Politely explain that you couldn't find relevant failure records for this specific query. 

Ask them to provide more details like:
- Specific failure codes
- Station numbers  
- Part numbers
- Error descriptions

Keep it conversational and helpful, no quotation marks."""
        
    else:
        # Casual conversation
        prompt = f"""You are a friendly manufacturing support assistant having a casual conversation.

The user said: {query}

Respond naturally and briefly. Let them know you're here to help with manufacturing failures, defects, part issues, and failure codes.

IMPORTANT: Do not put your response in quotation marks. Just respond naturally as if you're talking to them directly."""
    
    for token in llm.stream(prompt):
        yield token