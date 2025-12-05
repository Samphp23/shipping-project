import io
import os
import sys
import time
import json
import logging
import datetime
import argparse
import subprocess
import numpy as np
import pandas as pd
import pyarrow as pa
from io import StringIO
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path):
    try:
        with open(config_path) as f:
            config = json.load(f)
        required_keys = ["conn_string", "blob_container_name", "account_url","df_columns","destination_container_name"]
        for k in required_keys:
            if k not in config:
                raise KeyError(f"Missing required config key: {k}")
        return config
    except Exception as e:
        raise Exception(f"Error loading config file: {e}")  

def calculate_cbm(row):
    try:
        if pd.notna(row["CBM"]) and float(row["CBM"]) > 0:
            return row["CBM"]
        return (float(row["L_m"]) * float(row["Width_m"]) * float(row["Height_m"]))
    except Exception as e:
        logging.warning(f"CBM calculation failed for row {row.name}: {e}")
        return np.nan

def get_final_cbm(row):
    try:
        if pd.notna(row["CBM"]) and float(row["CBM"]) > 0:
            return row["CBM"]
        if pd.notna(row["Approx_CBM"]):
            return row["Approx_CBM"]
        if pd.notna(row["Avg_CBM"]):
            return row["Avg_CBM"]
        return 0
    except Exception as e:
        logging.warning(f"Final CBM fallback failed for row {row.name}: {e}")
        return 0

def get_final_weigth(row):
    try:
        if pd.notna(row["Weight"]) and float(row["Weight"]) > 0:
            return row["Weight"]
        if pd.notna(row["Weight_kg"]):
            return row["Weight_kg"]
        if pd.notna(row["Avg_Weight"]):
            return row["Avg_Weight"]
        return 0
    except Exception as e:
        logging.warning(f"Final Weight fallback failed for row {row.name}: {e}")
        return 0

def calculate_delivery_date(row):
    try:
        if pd.notna(row["DeliveryDate"]):
            return row["DeliveryDate"].date()
        if pd.notna(row["ProductionDate"]):
            return (row["ProductionDate"] + pd.Timedelta(days=5)).date()
        return (pd.Timestamp.today() + pd.Timedelta(days=30)).date()
    except Exception as e:
        logging.warning(f"DeliveryDate calculation failed for row {row.name}: {e}")
        return pd.Timestamp.today() + pd.Timedelta(days=30)

def connect_source_blob(conn_string, container_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        return blob_service_client.get_container_client(container_name)
    except Exception as e:
        raise Exception(f"Source_blob connection error: {e}")

def connect_destination_blob(conn_string, container_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        return blob_service_client.get_container_client(container_name)
    except Exception as e:
        raise Exception(f"Destination_blob connection error: {e}")

def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination_blob", required=True, help="Blob path passed from orchestration.py")
    args = parser.parse_args()
    destination_blob = args.destination_blob  #customer/kia/kia/filename_timestamp.csv

    config_path = os.path.join(base_dir, "kia_config.json")
    config = load_config(config_path)
    source_client = connect_source_blob(config["conn_string"], config["blob_container_name"])
    destination_client = connect_destination_blob(config["conn_string"], config["destination_container_name"])
    req_columns = config["df_columns"]

    try:
        data = source_client.download_blob(destination_blob).readall()
        df = pd.read_csv(StringIO(data.decode("utf-8")))
        model_blob = "common/model_map.csv"
        avg_blob = "common/avg_map.csv"
        source_data = source_client.download_blob(model_blob).readall()
        avg_data = source_client.download_blob(avg_blob).readall()
        model_map = pd.read_csv(StringIO(source_data.decode("utf-8")))
        avg_map = pd.read_csv(StringIO(avg_data.decode("utf-8")))
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        raise

    try:
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df = df[req_columns]
        df = df.dropna(subset=['Units'])
        df["Model"] = df["Model"].str.upper()
        df["Sender"] = df["Sender"].str.upper()
    except Exception as e:
        logging.error(f"Error cleaning dataframe: {e}")
        raise

    try: 
        df["DeliveryDate"]   = pd.to_datetime(df["DeliveryDate"], errors="coerce")
        df["ProductionDate"] = pd.to_datetime(df["ProductionDate"], errors="coerce")
    except Exception as e:
        logging.error(f"Error converting datatypes: {e}")
        raise

    try:
        df["CBM"] = df.apply(calculate_cbm, axis=1).round(2)
        df = df.merge(model_map, on=["Sender", "Model"], how="left")
        df = df.merge(avg_map, on="Model", how="left")
        df["CBM"] = df.apply(get_final_cbm, axis=1).round(2)
        df["Weight"] = df.apply(get_final_weigth, axis=1).round(2)
    except Exception as e:
        logging.error(f"Error apply and merging lookup tables: {e}")
        raise

    try:
        df["DeliveryDate"] = df.apply(calculate_delivery_date, axis=1)
        df["Result_CBM"] = (df["Units"] * df["CBM"]).round(2)
    except Exception as e:
        logging.error(f"Error mapping date and calculating Result_CBM: {e}")
        raise

    try:
        portmap_blob = "common/port_mapping.csv"
        portmap_data = source_client.download_blob(portmap_blob).readall()
        df_port = pd.read_csv(StringIO(portmap_data.decode("utf-8")))
        df = df.merge(df_port, left_on="Load_Port", right_on="POL", how="left")
        df['Trade'] = df['POL_continent']+"-"+df["POD_continent"]
        df = df.drop(columns=[]) 
        df = df.drop(columns=[
            "Length_mm", "Width_mm", "Height_mm",
            "Weight_kg", "Approx_CBM", "Avg_Weight",
            "Avg_CBM", "ProductionDate", "ArrivalDate",
            "POL_abr","POD_abr","POL_continent","POD_continent",
            "GateIn","L_m","Width_m","Height_m","CMB" ""
        ], errors="ignore")
    except Exception as e:
        logging.error(f"Error dropping columns: {e}")
        raise
        
    try:
        filename = os.path.basename(destination_blob).split('_')[0] + "_" + "silver_layer"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        silver_blob = f"kia/{filename}_{timestamp}.parquet"  #2nd container
        buffer = io.BytesIO() 
        table = pa.Table.from_pandas(df,preserve_index=False)
        pq.write_table(table, buffer,compression="snappy")
        buffer.seek(0)
        blob_client = destination_client.get_blob_client(silver_blob)
        blob_client.upload_blob(buffer, overwrite=True)
    except Exception as e:
        raise Exception(f"Error while creating file to silver folder: {e}")

    try:
        filename = os.path.basename(destination_blob)
        backup_blob = f"{config['backup_path']}{filename}_backup"
        print(backup_blob)
        backup_client = source_client.get_blob_client(backup_blob)
        source_blob_url = f"{config['account_url']}{config['blob_container_name']}/{destination_blob}"
        backup_client.start_copy_from_url(source_blob_url)
    except Exception as e:
        raise Exception(f"Error while creating file to backup: {e}")

    try:
        source_client.delete_blob(destination_blob)
        logging.info(f"Deleted destination_blob {destination_blob}")
    except Exception as e:
        raise Exception(f"Error while deleting destination_blob: {e}")

    elapsed_time = time.time() - start_time  
    logging.info(f"Silver_layer script completed in {elapsed_time:.2f} seconds")

    try:
        kia_path = os.path.join(base_dir,"gold_kia.py")
        logging.info("Running gold_kia.py...")
        subprocess.run([
            sys.executable, kia_path, "--silver_blob",silver_blob],check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Script execution failed: {e}")
        raise
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"ERROR OCCURRED: {e}", exc_info=True)
        sys.exit(1)