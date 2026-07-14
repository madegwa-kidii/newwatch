from __future__ import annotations

from typing import Optional

import lancedb
import numpy as np


class RecognitionService:
    """
    Handles all employee face recognition operations using LanceDB.

    Responsibilities:
    - Open/create the vector database
    - Add employees
    - Remove employees
    - Search for nearest employee
    """

    def __init__(
        self,
        db_path: str = "lancedb",
        table_name: str = "employees",
        threshold: float = 0.45,
    ):
        self.threshold = threshold

        self.db = lancedb.connect(db_path)

        table_names = self.db.table_names()

        if table_name in table_names:
            self.table = self.db.open_table(table_name)

        else:
            self.table = self.db.create_table(
                table_name,
                data=[
                    {
                        "employee_id": "",
                        "name": "",
                        "embedding": np.zeros(512, dtype=np.float32).tolist(),
                    }
                ],
                mode="overwrite",
            )

            # Remove placeholder row
            self.table.delete("employee_id == ''")

    # --------------------------------------------------

    def add_employee(
        self,
        employee_id: str,
        name: str,
        embedding: np.ndarray,
    ):

        self.table.add(
            [
                {
                    "employee_id": employee_id,
                    "name": name,
                    "embedding": embedding.astype(np.float32).tolist(),
                }
            ]
        )

    # --------------------------------------------------

    def remove_employee(self, employee_id: str):

        self.table.delete(f"employee_id == '{employee_id}'")

    # --------------------------------------------------

    def list_employees(self):

        return self.table.to_pandas()

    # --------------------------------------------------

    def recognize(
        self,
        embedding: np.ndarray,
    ) -> Optional[dict]:
        """
        Returns the closest employee.

        Returns None if the closest distance is above threshold.
        """

        results = (
            self.table.search(
                embedding.astype(np.float32).tolist()
            )
            .limit(1)
            .to_list()
        )

        if not results:
            return None

        match = results[0]

        distance = match["_distance"]

        if distance > self.threshold:
            return None

        return {
            "employee_id": match["employee_id"],
            "name": match["name"],
            "distance": distance,
        }