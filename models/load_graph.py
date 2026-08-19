"""
models/load_graph.py

Step 4b: data/graph_triples.json -> Neo4j Aura

Set in .env:
    NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=<your-password>

Run (after models/extract_graph.py):
    python models/load_graph.py
"""
import json
import os
import pathlib
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import GRAPH_TRIPLES_PATH as TRIPLES_PATH


def load_triples():
    if not TRIPLES_PATH.exists():
        raise FileNotFoundError(f"Could not find {TRIPLES_PATH}. Run models/extract_graph.py first.")
    return json.loads(TRIPLES_PATH.read_text(encoding="utf-8"))


def ensure_constraints(session):
    session.run(
        "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
    )


def load_record(session, record: dict) -> tuple[int, int]:
    source = {
        "chunk_id": record["chunk_id"],
        "section": record.get("section"),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "doi": record.get("doi"),
    }

    entities_skipped = 0
    relationships_skipped = 0

    for entity in record["entities"]:
        name = entity.get("name")
        etype = entity.get("type")
        if not name or not etype:
            entities_skipped += 1
            continue
        session.run("MERGE (e:Entity {name: $name, type: $type})", name=name, type=etype)

    for rel in record["relationships"]:
        src, tgt, relation = rel.get("source"), rel.get("target"), rel.get("relation")
        if not src or not tgt or not relation:
            relationships_skipped += 1
            continue
        session.run(
            """
            MERGE (a:Entity {name: $source})
            MERGE (b:Entity {name: $target})
            MERGE (a)-[r:RELATION {relation: $relation, chunk_id: $chunk_id}]->(b)
            SET r.section = $section,
                r.page_start = $page_start,
                r.page_end = $page_end,
                r.doi = $doi
            """,
            source=src, target=tgt, relation=relation,
            **source,
        )

    return entities_skipped, relationships_skipped


def main():
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([uri, user, password]):
        raise RuntimeError("NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD must be set in .env")

    records = load_triples()
    print(f"Loaded {len(records)} chunk records from {TRIPLES_PATH}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as session:
            print("Ensuring constraints ...")
            ensure_constraints(session)

            print("Loading entities and relationships ...")
            total_entities_skipped = 0
            total_relationships_skipped = 0

            for i, record in enumerate(records, start=1):
                e_skipped, r_skipped = load_record(session, record)
                total_entities_skipped += e_skipped
                total_relationships_skipped += r_skipped

                if i % 20 == 0 or i == len(records):
                    print(f"  {i}/{len(records)} chunk records loaded")

            if total_entities_skipped or total_relationships_skipped:
                print(f"\n(Skipped {total_entities_skipped} malformed entities and "
                      f"{total_relationships_skipped} malformed relationships from the source JSON.)")

            counts = session.run("MATCH (e:Entity) RETURN count(e) AS entities").single()
            rel_counts = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS relationships").single()

        print(f"\nGraph loaded: {counts['entities']} entities, {rel_counts['relationships']} relationships")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
