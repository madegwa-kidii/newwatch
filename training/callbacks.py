import numpy as np


class RegistrationCallback:
    """
    Stores employee face embeddings in LanceDB.
    """

    def __init__(self, database):
        self.database = database

    def save_embedding(
        self,
        employee_name: str,
        embedding: np.ndarray,
    ) -> None:
        """
        Save one ArcFace embedding.
        """

        self.database.add_embedding(
            employee_name=employee_name,
            embedding=embedding,
        )

        print(f"✓ Saved embedding for {employee_name}")