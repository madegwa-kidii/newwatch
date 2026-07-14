from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa


DB_PATH = Path("data/database")

DB_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

db = lancedb.connect(DB_PATH)


schema = pa.schema([
    pa.field("name", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 512)),
])


try:
    table = db.open_table("employees")

except Exception:

    table = db.create_table(
        "employees",
        schema=schema,
    )


class EmployeeDatabase:

    def __init__(self):
        self.table = table

    def add_employee(self, name: str, embedding):

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).tolist()

        self.table.add([
            {
                "name": name,
                "embedding": embedding,
            }
        ])

    def search(self, embedding, threshold=0.45):

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        result = (
            self.table
            .search(
                embedding.tolist(),
                vector_column_name="embedding",
            )
            .metric("cosine")
            .limit(1)
            .to_list()
        )

        if len(result) == 0:
            return None

        person = result[0]

        if person["_distance"] > threshold:
            return None

        return person

    def list_employees(self):
        return (
            self.table
            .search()
            .limit(100000)
            .to_list()
        )

    def delete_employee(self, name):

        self.table.delete(f"name = '{name}'")