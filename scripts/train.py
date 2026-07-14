from training.pipeline import TrainingPipeline
from database.lancedb import EmployeeDatabase


def main():

    trainer = TrainingPipeline()
    trainer.train()

    db = EmployeeDatabase()
    records = db.list_employees()

    print("\nDatabase contains", len(records), "embeddings")

    for i, r in enumerate(records):
        emb = r["embedding"]

        print(f"\nRecord {i}")
        print("Name:", r["name"])
        print("Length:", len(emb))
        print("First 10:", emb[:10])


if __name__ == "__main__":
    main()