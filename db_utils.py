from pymongo import MongoClient
from sqlalchemy import create_engine
from pyhive import hive

def connect_to_db(db_type, config):
    """Connect to SQL, NoSQL, MongoDB, Apache, or Cloud databases."""
    try:
        if db_type == "SQL":
            engine = create_engine(
                f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
            )
            return engine.connect()

        elif db_type == "MongoDB":
            client = MongoClient(config['host'], int(config['port']))
            return client[config['database']]

        elif db_type == "Apache":
            conn = hive.Connection(
                host=config['host'],
                port=int(config['port']),
                username=config['user'],
                password=config['password'],
                database=config['database']
            )
            return conn

        elif db_type == "Cloud":
            engine = create_engine(config['cloud_url'])
            return engine.connect()

        else:
            return None

    except Exception as e:
        print(f"⚠️ Database connection error: {str(e)}")
        return None
