# sparql_query.py
from rdflib import Graph

def query_rdf_graph(query_string):
    """
    Execute a SPARQL query on the RDF knowledge graph and return the results.
    """
    # Load the RDF graph (replace 'data_ontology.ttl' with the actual path to your RDF file)
    g = Graph()
    g.parse("data_ontology.ttl", format="turtle")

    # Execute the SPARQL query
    results = g.query(query_string)
    
    # Collect results and return them
    collected_results = []
    for row in results:
        collected_results.append(row)
    
    return collected_results


def retrieve_audit_data(query):
    """
    Query the RDF graph to retrieve datasets, audits, scope, and dataset links based on the user's query.
    """
    sparql_query = f"""
    PREFIX ex: <http://example.org/ontology/>
    SELECT ?dataset ?audit ?scope ?link
    WHERE {{
      ?dataset ex:hasCategory ?category .
      ?category ex:includesAudit ?audit .
      ?audit ex:hasScope ?scope .
      ?dataset ex:hasLink ?link .
      FILTER(CONTAINS(LCASE(?scope), "{query.lower()}"))
    }}
    """
    return query_rdf_graph(sparql_query)
