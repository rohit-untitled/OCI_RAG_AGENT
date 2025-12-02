import oci
import os

# Base project path (automatically detected)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder where all OCI downloads will be stored
LOCAL_DOWNLOAD_DIR = os.path.join(BASE_DIR, "data", "downloads")

OCI_NAMESPACE = "idnutrlj05lw"
OCI_BUCKET_NAME = "AnonymizedProjects"

def download_folder_from_oci(prefix: str):
    """
    Downloads all files from an OCI Object Storage folder (prefix) 
    into a local directory inside the project.
    """
    try:
        # Initialize OCI client
        config = oci.config.from_file()
        object_storage_client = oci.object_storage.ObjectStorageClient(config)

        # Get list of all objects under the prefix
        objects = object_storage_client.list_objects(
            OCI_NAMESPACE, OCI_BUCKET_NAME, prefix=prefix
        ).data.objects

        if not objects:
            return {"status": "error", "message": f"No files found for prefix '{prefix}'"}

        # Ensure local folder exists
        os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

        # Download each object
        for obj in objects:
            file_name = obj.name
            if file_name.endswith("/"):
                continue  # skip folder markers

            print(f"Downloading {file_name}...")

            local_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, file_name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            response = object_storage_client.get_object(
                OCI_NAMESPACE, OCI_BUCKET_NAME, file_name
            )

            with open(local_file_path, "wb") as f:
                f.write(response.data.content)

            print(f"✅ Downloaded {file_name}")

        return {
            "status": "success",
            "message": f"All files from '{prefix}' downloaded to {LOCAL_DOWNLOAD_DIR}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
