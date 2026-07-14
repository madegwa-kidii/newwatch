from database.lancedb import EmployeeDatabase


class DatabaseHandler:
    """
    Compatibility layer between the Hailo training pipeline
    and our LanceDB backend.
    """

    def __init__(self):
        self.db = EmployeeDatabase()

    def insert_embedding(self, name, embedding):
        """
        Store one embedding in LanceDB.
        """
        self.db.insert_embedding(
            name=name,
            embedding=embedding,
        )

    def search_embedding(self, embedding):
        """
        Search the nearest embedding.
        """
        return self.db.search(embedding)

    def delete_employee(self, name):
        self.db.delete_employee(name)

    def list_employees(self):
        return self.db.list_employees()