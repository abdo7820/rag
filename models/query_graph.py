"""
models/query_graph.py

Quick sanity check: look up an entity in Neo4j and print what it's connected to.

Run:
    python models/query_graph.py "Cirrhosis"
"""
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase


def print_neighbors(session, entity_name: str):
    result = session.run(
        """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($name)
           OR toLower(b.name) CONTAINS toLower($name)
        RETURN a.name AS source, r.relation AS relation, b.name AS target,
               r.section AS section, r.page_start AS page_start,
               r.page_end AS page_end, r.doi AS doi
        LIMIT 25
        """,
        name=entity_name,
    )

    rows = list(result)
    if not rows:
        print(f"No relationships found for '{entity_name}'.")
        return

    for row in rows:
        pages = row["page_start"]
        if row["page_end"] and row["page_end"] != row["page_start"]:
            pages = f"{row['page_start']}-{row['page_end']}"

        print(f"{row['source']} -[{row['relation']}]-> {row['target']}")
        print(f"    section={row['section']} | pages={pages} | doi={row['doi']}")


def main():
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([uri, user, password]):
        raise RuntimeError("NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD must be set in .env")

    query = " ".join(sys.argv[1:]) or "Cirrhosis"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            print(f"Entity: {query}\n")
            print_neighbors(session, query)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
