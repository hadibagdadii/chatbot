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
2. ALWAYS cite the TOTAL database records first, then mention analyzed subset
3. Be direct and conversational - no formal headers like "Analysis Report"
4. Always cite exact counts from the database stats
5. Give practical, actionable advice

Respond in a helpful, direct tone. Start with the database totals, then provide recommendations."""

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
    
    # Check for code patterns
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
    """Format the aggregated data with EXACT database statistics"""
    
    parts = []
    
    # Add query context
    parts.append(f"USER QUERY: {agg.get('query_context', 'N/A')}")
    
    # Add EXACT database statistics (most important!)
    db_stats = agg.get('database_stats', {})
    if db_stats:
        parts.append("\n---EXACT DATABASE STATISTICS---")
        for key, value in db_stats.items():
            if 'total' in key:
                if 'part_' in key:
                    part_num = key.split('_')[1]
                    parts.append(f"\nTOTAL RECORDS FOR PART {part_num}: {value}")
                elif 'serial_' in key:
                    serial = key.split('_')[1]
                    parts.append(f"\nTOTAL RECORDS FOR SERIAL {serial}: {value}")
                elif 'failure_' in key:
                    code = key.split('_')[1]
                    parts.append(f"\nTOTAL RECORDS FOR FAILURE CODE {code}: {value}")
            elif '_failures' in key:
                part_num = key.split('_')[1]
                parts.append(f"\nFAILURE CODES FOR PART {part_num}:")
                for code, count in value[:5]:  # Top 5
                    parts.append(f"  • {code}: {count}x")
            elif '_materials' in key:
                part_num = key.split('_')[1]
                parts.append(f"\nMATERIALS USED FOR PART {part_num}:")
                for mat_code, mat_desc, part_class, count in value[:5]:  # Top 5
                    parts.append(f"  • {mat_code} - {mat_desc} ({part_class}): {count}x")
            elif '_recurring_serials' in key:
                part_num = key.split('_')[1]
                if value:
                    parts.append(f"\nRECURRING SERIALS FOR PART {part_num}:")
                    for serial, count in value:
                        parts.append(f"  • Serial {serial}: {count}x")
    
    parts.append(f"\nSEMANTIC SEARCH ANALYZED: {agg.get('retrieved_count', 0)} most relevant records")
    
    parts.append("\n---DATA FROM SEMANTIC SEARCH---\n")
    
    # Action codes from semantic search
    parts.append("MOST COMMON ACTION CODES (in analyzed subset):")
    action_codes = agg.get('action_codes', [])
    if action_codes and isinstance(action_codes[0], dict) and 'code' in action_codes[0]:
        for item in action_codes:
            parts.append(f"  • {item['code']} (seen {item['count']}x in analyzed records)")
    else:
        parts.append("  • No action codes found")
    
    # Failure codes from semantic search
    failure_codes = agg.get('failure_codes', [])
    if failure_codes and isinstance(failure_codes[0], dict) and 'code' in failure_codes[0]:
        parts.append("\nFAILURE CODES (in analyzed subset):")
        for item in failure_codes:
            parts.append(f"  • {item['code']} (seen {item['count']}x in analyzed records)")
    
    # Materials from semantic search
    parts.append("\nMATERIALS (in analyzed subset):")
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
            parts.append(f"    Seen {count}x in analyzed records")
    else:
        parts.append("  • No materials recorded")
    
    # Recurring serials from semantic search
    recurring = agg.get('recurring_serials', [])
    if recurring and isinstance(recurring[0], dict) and 'serial' in recurring[0]:
        parts.append("\nRECURRING SERIALS (in analyzed subset):")
        for item in recurring:
            parts.append(f"  • Serial {item['serial']} appeared {item['count']}x")
    
    # Stations
    stations = agg.get('common_stations', [])
    if stations and len(stations) > 0:
        parts.append("\nCOMMON STATIONS:")
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

Based on the data above, answer this question directly: {query}

IMPORTANT: 
- Start by citing the TOTAL database records (e.g., "I found 63 total records for part 10003939 in the database")
- Then mention you analyzed the most relevant subset
- Provide practical recommendations based on the patterns
- Be conversational and direct - no formal headers

Give specific advice including:
- What the failure typically indicates
- What action to take (cite the action code)
- What part to replace (cite the material code and description)
- Any recurring patterns (serials, stations)"""
        
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

IMPORTANT: Do not put your response in quotation marks. Just respond naturally."""
    
    for token in llm.stream(prompt):
        yield token