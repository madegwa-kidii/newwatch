from pathlib import Path

from database.lancedb import EmployeeDatabase
from face.embedding_pipeline import EmbeddingPipeline


class TrainingPipeline:

    def __init__(
        self,
        dataset_dir="data/employees",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.db = EmployeeDatabase()
        self.pipeline = EmbeddingPipeline()

    def train(self):

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"{self.dataset_dir} does not exist."
            )

        total_people = 0
        total_images = 0

        print("\n========== TRAINING ==========\n")

        for person_dir in sorted(self.dataset_dir.iterdir()):

            if not person_dir.is_dir():
                continue

            name = person_dir.name

            print(f"Employee: {name}")

            image_count = 0

            image_files = []

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                image_files.extend(person_dir.glob(ext))

            for image_path in sorted(image_files):

                print(f"  Processing {image_path.name}...")

                embedding = self.pipeline.get_embedding(image_path)

                if embedding is None:
                    print("    No face found.")
                    continue

                self.db.add_employee(
                    name=name,
                    embedding=embedding,
                )

                image_count += 1
                total_images += 1

            print(f"  Stored {image_count} embeddings\n")

            total_people += 1

        print("========== COMPLETE ==========")
        print(f"Employees : {total_people}")
        print(f"Embeddings: {total_images}")