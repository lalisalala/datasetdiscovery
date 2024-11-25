# utils.py

# Mapping of query types to keywords and SPARQL query templates
QUERY_TYPE_MAP = {
    "structure": {
        "keywords": ["structure", "columns", "fields", "data layout", "schema"],
        "sparql_template": """
            PREFIX ex: <http://example.org/ontology/>
            SELECT ?column ?label ?dataType
            WHERE {
              ?column a ex:Column ;
                      rdfs:label ?label ;
                      ex:dataType ?dataType .
            }
        """
    },
    "general": {
        "keywords": [],  # Optional default or fallback for general queries
        "sparql_template": """
            PREFIX ex: <http://example.org/ontology/>
            SELECT ?dataset ?title ?summary ?link ?row ?property ?value
            WHERE {
              ?dataset ex:hasTitle ?title .
              ?dataset ex:hasSummary ?summary .
              ?dataset ex:hasLink ?link .
              ?row ex:partOf ?dataset .
              ?row ?property ?value .
            }
        """
    }
}

def detect_query_type(query: str) -> str:
    """
    Detect the type of query based on keywords and return the corresponding SPARQL template.

    Args:
        query (str): The user query string.

    Returns:
        Tuple[str, str]: Detected query type and SPARQL template, or "general" if no type detected.
    """
    query_lower = query.lower()
    
    # Check each query type in QUERY_TYPE_MAP for keyword matches
    for query_type, config in QUERY_TYPE_MAP.items():
        if any(keyword in query_lower for keyword in config["keywords"]):
            return query_type, config["sparql_template"]
    
    # Default to general query if no specific type is matched
    return "general", QUERY_TYPE_MAP["general"]["sparql_template"]
