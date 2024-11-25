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
    Returns all datasets and rows, allowing for broader analysis.
    """
    sparql_query = """
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
    return query_rdf_graph(sparql_query)

def build_dynamic_sparql_query(query_type):
    """Generate a SPARQL query dynamically based on query type."""
    if query_type == "structure":
        return """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?column ?label ?dataType
        WHERE {
            ?column a ex:Column ;
                    rdfs:label ?label ;
                    ex:dataType ?dataType .
        }
        """
    elif query_type == "content":
        return """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?dataset ?title ?summary ?link ?row ?property ?value
        WHERE {
            ?dataset ex:hasTitle ?title ;
                     ex:hasSummary ?summary ;
                     ex:hasLink ?link .
            ?row ex:partOf ?dataset ;
                 ?property ?value .
        }
        """
    else:  # Fallback for general queries
        return """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o .
        }
        LIMIT 50
        """
def classify_query(query):
    """Classify query type based on keywords."""
    structure_keywords = ["structure", "columns", "fields", "schema"]
    content_keywords = ["summary", "data", "values", "rows"]

    if any(keyword in query.lower() for keyword in structure_keywords):
        return "structure"
    elif any(keyword in query.lower() for keyword in content_keywords):
        return "content"
    else:
        return "general"