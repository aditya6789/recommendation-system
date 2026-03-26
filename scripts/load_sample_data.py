"""Load sample MovieLens-like data into PostgreSQL."""

from sqlalchemy import select

from app.db.database import Base, SessionLocal, engine
from app.models import Item, Rating, User
from app.services.recommendation_service import recommendation_service


def seed_data() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.execute(select(User.id)).first():
            print("Data already exists. Skipping seed.")
            return

        users = [
            User(name="Alice", email="alice@example.com"),
            User(name="Bob", email="bob@example.com"),
            User(name="Charlie", email="charlie@example.com"),
            User(name="Diana", email="diana@example.com"),
        ]
        items = [
            Item(title="The Matrix", genre="Sci-Fi", tags="ai cyberspace action", description="Virtual reality rebellion."),
            Item(title="Inception", genre="Sci-Fi", tags="dream mind-bending thriller", description="Dream layers and heist."),
            Item(title="Interstellar", genre="Sci-Fi", tags="space time drama", description="Space travel to save humanity."),
            Item(title="Titanic", genre="Romance", tags="love tragedy ship", description="Romance on doomed voyage."),
            Item(title="The Dark Knight", genre="Action", tags="hero crime gotham", description="Batman vs Joker."),
            Item(title="Toy Story", genre="Animation", tags="family toys comedy", description="Toys come to life."),
            Item(title="Avengers: Endgame", genre="Action", tags="superhero marvel epic", description="Final showdown."),
            Item(title="La La Land", genre="Musical", tags="music romance hollywood", description="Artists chasing dreams."),
        ]
        db.add_all(users + items)
        db.commit()

        ratings = [
            Rating(user_id=1, item_id=1, rating=5.0),
            Rating(user_id=1, item_id=2, rating=4.5),
            Rating(user_id=1, item_id=3, rating=4.0),
            Rating(user_id=2, item_id=1, rating=4.0),
            Rating(user_id=2, item_id=5, rating=5.0),
            Rating(user_id=2, item_id=7, rating=4.5),
            Rating(user_id=3, item_id=4, rating=4.5),
            Rating(user_id=3, item_id=8, rating=4.0),
            Rating(user_id=3, item_id=6, rating=4.5),
            Rating(user_id=4, item_id=2, rating=5.0),
            Rating(user_id=4, item_id=3, rating=4.5),
            Rating(user_id=4, item_id=5, rating=4.0),
        ]
        db.add_all(ratings)
        db.commit()

        recommendation_service.rebuild(db)
        print("Sample data loaded successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
