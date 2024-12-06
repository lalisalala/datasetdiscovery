import logging
import re
from rdflib import Graph

logger = logging.getLogger(__name__)

def query_rdf_graph(query_string):
    """
    Execute a SPARQL query on the RDF knowledge graph and return the results.
    """
    try:
        # Load the RDF graph (replace 'metadata_ontology.ttl' with your actual RDF file path)
        g = Graph()
        g.parse("metadata_ontology.ttl", format="turtle")

        # Log the query for debugging
        logger.debug(f"Executing SPARQL Query:\n{query_string}")

        # Execute the SPARQL query
        results = g.query(query_string)

        # Collect results and return them
        collected_results = []
        for row in results:
            collected_results.append(row)

        return collected_results

    except Exception as e:
        logger.error(f"Error executing SPARQL query: {e}")
        return []


def retrieve_metadata(query=None):
    """
    Query the RDF graph to retrieve dataset metadata such as title, summary, links, and topics.
    If a query is provided, additional SPARQL filters will be applied.
    """
    sparql_query = """
    PREFIX ex: <http://example.org/ontology/>
    PREFIX schema: <https://schema.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?dataset ?title ?summary ?publisher ?topic ?link ?format
    WHERE {
      ?dataset a ex:Dataset ;
               rdfs:label ?title ;
               ex:summary ?summary ;
               ex:publisher ?publisher ;
               ex:fileFormat ?format .
      OPTIONAL { ?dataset ex:hasTopic/skos:prefLabel ?topic . }
      OPTIONAL { ?dataset schema:url ?link . }
    """
    
    # Add filters based on the user query
    if query:
        # Example: Searching by topic or title
        sparql_query += f"""
          FILTER(
            CONTAINS(LCASE(?title), LCASE("{query}")) ||
            CONTAINS(LCASE(?topic), LCASE("{query}"))
          )
        """
    
    # Close the SPARQL query
    sparql_query += """
    }
    """
    
    # Log the query for debugging
    logger.debug(f"Generated SPARQL Query:\n{sparql_query}")
    
    return query_rdf_graph(sparql_query)



def build_dynamic_sparql_query_optimized(query_type, user_query=None):
    """
    Generate a SPARQL query dynamically based on query type and user intent.
    Focused on known metadata fields for efficiency.

    Args:
        query_type (str): The type of query (e.g., 'metadata', 'filter by topic', etc.).
        user_query (str): Optional user query to guide the SPARQL generation.

    Returns:
        str: A SPARQL query string tailored to the query type.
    """
    # Define SPARQL templates
    templates = {
        "all_metadata": """
        PREFIX ex: <http://example.org/ontology/>
        PREFIX schema: <https://schema.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?dataset ?title ?summary ?publisher ?topic ?link ?format
        WHERE {
          ?dataset a ex:Dataset ;
                   rdfs:label ?title ;
                   ex:summary ?summary ;
                   ex:publisher ?publisher ;
                   ex:fileFormat ?format .
          OPTIONAL { ?dataset ex:hasTopic/skos:prefLabel ?topic . }
          OPTIONAL { ?dataset schema:url ?link . }
        }
        """,
        "filter_by_topic": """
        PREFIX ex: <http://example.org/ontology/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?dataset ?title ?summary ?link
        WHERE {
          ?dataset a ex:Dataset ;
                   rdfs:label ?title ;
                   ex:summary ?summary ;
                   ex:hasTopic/skos:prefLabel ?topic ;
                   schema:url ?link .
          FILTER(CONTAINS(LCASE(?topic), LCASE("{topic}")))
        }
        """,
        "filter_by_publisher": """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?dataset ?title ?summary ?link
        WHERE {
          ?dataset a ex:Dataset ;
                   rdfs:label ?title ;
                   ex:summary ?summary ;
                   ex:publisher ?publisher ;
                   schema:url ?link .
          FILTER(CONTAINS(LCASE(?publisher), LCASE("{publisher}")))
        }
        """,
        "filter_by_topic_and_publisher": """
        PREFIX ex: <http://example.org/ontology/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?dataset ?title ?summary ?link
        WHERE {
          ?dataset a ex:Dataset ;
                   rdfs:label ?title ;
                   ex:summary ?summary ;
                   ex:hasTopic/skos:prefLabel ?topic ;
                   ex:publisher ?publisher ;
                   schema:url ?link .
          FILTER(
              CONTAINS(LCASE(?topic), LCASE("{topic}")) &&
              CONTAINS(LCASE(?publisher), LCASE("{publisher}"))
          )
        }
        """
    }

    # Determine the query to use based on the query type
    if query_type == "all_metadata":
        return templates["all_metadata"]

    if query_type == "filter_by_topic" and user_query:
        topic = extract_keyword(user_query, context="topic")
        return templates["filter_by_topic"].format(topic=topic)

    if query_type == "filter_by_publisher" and user_query:
        publisher = extract_keyword(user_query, context="publisher")
        return templates["filter_by_publisher"].format(publisher=publisher)

    if query_type == "filter_by_topic_and_publisher" and user_query:
        topic = extract_keyword(user_query, context="topic")
        publisher = extract_keyword(user_query, context="publisher")
        return templates["filter_by_topic_and_publisher"].format(topic=topic, publisher=publisher)

    # Fallback: General metadata query
    return templates["all_metadata"]


def extract_keyword(user_query, context):
    """
    Extract keywords from the user query based on context.
    """
    if context == "topic":
        match = re.search(r"(topic|about|related to|theme)\s+(.*)", user_query, re.IGNORECASE)
    elif context == "publisher":
        match = re.search(r"(published by|from|organization|publisher)\s+(.*)", user_query, re.IGNORECASE)
    else:
        match = None

    if match:
        return match.group(2).strip().split()[0]  # Return the first word after the context
    return user_query  # Fallback to the full query


def classify_query(query):
    """
    Classify query type based on keywords.
    Focuses on metadata-related queries.
    """
    metadata_keywords = ["title", "summary", "publisher", "topic", "format", "link", "metadata"]

    if any(keyword in query.lower() for keyword in metadata_keywords):
        return "metadata"
    else:
        return "general"


def validate_sparql_query(query_string):
    """
    Validate the SPARQL query before execution.
    """
    required_prefixes = ["PREFIX ex:", "PREFIX schema:", "PREFIX skos:"]
    for prefix in required_prefixes:
        if prefix not in query_string:
            logging.warning(f"SPARQL query missing required prefix: {prefix}")
            return False
    return True
