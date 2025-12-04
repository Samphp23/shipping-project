import os
import sys
import time
import json
import logging
import argparse
import sqlalchemy
import urllib.parse
import pandas as pd
import pyarrow as pa
from io import BytesIO
from sqlalchemy import create_engine
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_config(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
        logging.info("Loaded configuration file successfully")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        raise

def load_dataframe_from_blob(conn_string, container_name, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        container_client = blob_service_client.get_container_client(container_name)
        data = container_client.download_blob(blob_name).readall()
        df = pd.read_parquet(BytesIO(data))
        logging.info(f"Loaded DataFrame from blob {blob_name} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Failed to read blob or load DataFrame: {e}")
        raise

def create_sqlalchemy_engine(username, password, server, database):
    try:
        quoted_pwd = urllib.parse.quote_plus(password)

        connection_string = (
            f"mssql+pyodbc://{username}:{quoted_pwd}@{server}.database.windows.net/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )

        engine = sqlalchemy.create_engine(connection_string)
        logging.info("SQLAlchemy engine created successfully")
        return engine

    except Exception as e:
        logging.error(f"Failed to create SQLAlchemy engine: {e}")
        raise

def dump_pol_tables(df, engine):
    try:
        unique_POL = df["POL"].unique()
        for k in unique_POL:
            df_POL = df[df["POL"] == k]
            df_POL.to_sql(f"kia_{k}", con=engine, if_exists="append", index=False, chunksize=1000)
            logging.info(f"Inserted {len(df_POL)} rows into table kia_{k}")
    except Exception as e:
        logging.error("Failed during POL table insert operations: %s", e)
        raise

def insert_summary_table(df, engine):
    try:
        agg_pol_segment = df.groupby(["POL", "Cargo_Segment"]).agg(
            total_units=pd.NamedAgg(column="Units", aggfunc="sum"),
            total_cbm=pd.NamedAgg(column="Result_CBM", aggfunc="sum"),
            shipment_count=pd.NamedAgg(column="Sender_ID", aggfunc="count")
        ).reset_index()

        agg_pol_segment["total_cbm"] = agg_pol_segment["total_cbm"].round(2)

        agg_pol_segment.to_sql("kia_summary", con=engine, if_exists="append", index=False, chunksize=1000)

        logging.info(f"Inserted {len(agg_pol_segment)} rows into table kia_summary")

    except Exception as e:
        logging.error("Failed during summary table insert operations: %s", e)
        raise

def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--silver_blob", required=True, help="Blob path to parquet file")
        args = parser.parse_args()

        config_path = os.path.join(base_dir, "kia_config.json")
        config = load_config(config_path)

        KEYVAULT_URL = "https://mlstudio1612843333.vault.azure.net/"
        credential = DefaultAzureCredential()
        kv_client = SecretClient(vault_url=KEYVAULT_URL, credential=credential)
        logging.info("Fetching secrets from KeyVault...")
        server = kv_client.get_secret("server").value
        database = kv_client.get_secret("database").value
        username = kv_client.get_secret("username").value
        password = kv_client.get_secret("password").value
        conn_string = kv_client.get_secret("conn-string").value
        logging.info("Secrets loaded successfully from KeyVault")

        df = load_dataframe_from_blob(conn_string,
                                      config["destination_container_name"],
                                      args.silver_blob)

        engine = create_sqlalchemy_engine(username, password, server, database)
        dump_pol_tables(df, engine)
        insert_summary_table(df, engine)
        logging.info(f"Gold pipeline completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error("ERROR OCCURRED IN MAIN: %s", e, exc_info=True)

    logging.info("SUCCESS: gold_layer script completed without errors.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("ERROR OCCURRED: %s", e, exc_info=True)
        sys.exit(1)
