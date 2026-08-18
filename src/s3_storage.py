import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# AWS Credentials setup (Uses env vars or defaults)
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "rag-agent-research-papers")

class S3StorageManager:
    def __init__(self, bucket_name: str = S3_BUCKET_NAME, region: str = AWS_REGION):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region)

    def create_bucket_if_not_exists(self):
        """Creates the S3 bucket if it doesn't already exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' already exists.")
        except ClientError:
            try:
                if AWS_REGION == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                    )
                print(f"Successfully created S3 bucket: {self.bucket_name}")
            except Exception as e:
                print(f"Failed to create S3 bucket: {e}")

    def upload_file(self, file_path: str, object_name: str = None) -> str:
        """Uploads a local PDF or document to S3 and returns the S3 URI."""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            s3_uri = f"s3://{self.bucket_name}/{object_name}"
            print(f"Uploaded {file_path} to {s3_uri}")
            return s3_uri
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
            return ""
        except NoCredentialsError:
            print("AWS credentials not available.")
            return ""

    def list_files(self) -> list[str]:
        """Lists all object keys in the bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

if __name__ == "__main__":
    storage = S3StorageManager()
    storage.create_bucket_if_not_exists()