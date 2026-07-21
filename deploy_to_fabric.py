"""
Deploy Environment and Spark Job Definition to Microsoft Fabric
================================================================
# Run: python deploy_to_fabric.py --workspace-id "da5ae027-82f7-4d2a-a757-00a39dc08406" --lakehouse-id "b7920828-a740-4c11-9e3b-4cdaa16e30ab"

This script deploys the ts-forecasting-env environment and e2e-ml-pipeline
Spark Job Definition to your Microsoft Fabric workspace using the Fabric REST API.

Prerequisites:
    - Azure CLI installed and authenticated (az login)
    - Fabric workspace created
    - Lakehouse created in the workspace

Usage:
    python deploy_to_fabric.py --workspace-id <YOUR_WORKSPACE_ID> --lakehouse-id <YOUR_LAKEHOUSE_ID>

Optional:
    --environment-name: Name for the environment (default: ts-forecasting-env)
    --job-name: Name for the Spark Job Definition (default: e2e-ml-pipeline-spark-job)
    --src-dir: Local source folder containing *.py files (default: ./src)
"""

import argparse
import base64
import json
import sys
import time
import requests
from pathlib import Path


class FabricDeployer:
    """Deploy items to Microsoft Fabric using the REST API."""
    
    FABRIC_API_BASE = "https://api.fabric.microsoft.com/v1"
    
    def __init__(self, workspace_id: str, lakehouse_id: str):
        """
        Initialize the deployer.
        
        Args:
            workspace_id: The Fabric workspace ID
            lakehouse_id: The lakehouse ID for default attachment
        """
        self.workspace_id = workspace_id
        self.lakehouse_id = lakehouse_id
        self.access_token = None
        
    def authenticate(self):
        """Get access token using Azure CLI."""
        print("🔐 Authenticating with Azure CLI...")
        try:
            import subprocess
            # Use shell=True on Windows to find az.cmd
            result = subprocess.run(
                ["az", "account", "get-access-token", "--resource", "https://api.fabric.microsoft.com"],
                capture_output=True,
                text=True,
                check=True,
                shell=True  # Required on Windows to find az.cmd
            )
            token_data = json.loads(result.stdout)
            self.access_token = token_data["accessToken"]
            print("✅ Authentication successful")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("Please run 'az login' first")
            print("\nTroubleshooting:")
            print("1. Verify Azure CLI is installed: az --version")
            print("2. Ensure you're logged in: az account show")
            sys.exit(1)
    
    def _get_headers(self, include_json_content_type: bool = True):
        """Get HTTP headers with authentication."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        if include_json_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _to_inline_base64(text: str) -> str:
        """Encode text content as base64 for Fabric definition parts."""
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    def _resolve_source_dir(self, src_dir: str) -> Path:
        """Resolve source folder and require main.py in that folder."""
        src_path = Path(src_dir)
        if not src_path.is_absolute():
            src_path = Path(__file__).parent / src_path

        if not src_path.exists() or not src_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {src_path}")

        if (src_path / "main.py").exists():
            return src_path

        raise FileNotFoundError(
            f"main.py not found in: {src_path}. Use --src-dir to point to folder containing main.py"
        )

    def _build_definition_parts(self, job_config: dict, src_path: Path):
        """Build Spark Job definition parts (config + local files)."""
        parts = []

        py_files = sorted(src_path.glob("*.py"))
        if not py_files:
            raise FileNotFoundError(f"No Python files found in: {src_path}")

        main_file = None
        lib_files = []
        for py_file in py_files:
            if py_file.name == "main.py":
                main_file = py_file
            else:
                lib_files.append(py_file)

        if main_file is None:
            raise FileNotFoundError(f"main.py not found in: {src_path}")

        with open(main_file, "r", encoding="utf-8") as f:
            main_content = f.read()
        parts.append(
            {
                "path": "Main/main.py",
                "payload": self._to_inline_base64(main_content),
                "payloadType": "InlineBase64"
            }
        )

        for lib_file in lib_files:
            with open(lib_file, "r", encoding="utf-8") as f:
                lib_content = f.read()
            parts.append(
                {
                    "path": f"Libs/{lib_file.name}",
                    "payload": self._to_inline_base64(lib_content),
                    "payloadType": "InlineBase64"
                }
            )

        print(f"✅ Packaged {len(py_files)} local files from: {src_path}")

        parts.insert(
            0,
            {
                "path": "SparkJobDefinitionV1.json",
                "payload": self._to_inline_base64(json.dumps(job_config)),
                "payloadType": "InlineBase64"
            }
        )
        return parts

    def _find_existing_item_id(self, display_name: str, item_type: str):
        """Find an existing workspace item by display name and type."""
        url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/items"

        while url:
            response = requests.get(url, headers=self._get_headers())
            if response.status_code != 200:
                return None

            data = response.json()
            items = data.get("value", [])
            for item in items:
                if item.get("displayName") == display_name and item.get("type") == item_type:
                    return item.get("id")

            continuation_uri = data.get("continuationUri")
            continuation_token = data.get("continuationToken")
            if continuation_uri:
                url = continuation_uri
            elif continuation_token:
                url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/items?continuationToken={continuation_token}"
            else:
                url = None

        return None

    def _upload_and_publish_environment(self, env_id: str, env_yml_content: str):
        """Upload environment YAML and publish the environment."""
        print("📝 Uploading environment.yml...")
        update_url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/environments/{env_id}/staging/libraries"

        multipart_files = {
            "environment": ("environment.yml", env_yml_content, "text/yaml")
        }
        update_response = requests.post(
            update_url,
            headers=self._get_headers(include_json_content_type=False),
            files=multipart_files,
        )
        if update_response.status_code in [200, 201, 202]:
            print("✅ Environment configuration uploaded")
            print("📢 Publishing environment...")
            publish_url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/environments/{env_id}/staging/publish"
            publish_response = requests.post(publish_url, headers=self._get_headers())

            if publish_response.status_code in [200, 202]:
                print("✅ Environment published successfully")
            else:
                print(f"⚠️ Publish status: {publish_response.status_code}")
        else:
            print(f"⚠️ Configuration upload status: {update_response.status_code}")
            print(update_response.text)

    def create_environment(self, name: str = "ts-forecasting-env") -> str:
        """
        Create a Fabric Environment with required packages.
        
        Args:
            name: Name of the environment
            
        Returns:
            Environment ID
        """
        print(f"\n📦 Creating Environment: {name}")
        
        # Read environment.yml
        env_file = Path(__file__).parent / "env" / "environment.yml"
        with open(env_file, 'r') as f:
            env_yml_content = f.read()
        
        # Create environment item
        url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/items"
        payload = {
            "displayName": name,
            "type": "Environment",
            "description": "Environment for time-series forecasting with SARIMAX"
        }
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        
        if response.status_code == 201:
            env_data = response.json()
            env_id = env_data["id"]
            print(f"✅ Environment created: {env_id}")
            self._upload_and_publish_environment(env_id, env_yml_content)
            
            return env_id
        elif response.status_code == 409:
            existing_env_id = self._find_existing_item_id(name, "Environment")
            if existing_env_id:
                print(f"ℹ️ Environment already exists, reusing: {existing_env_id}")
                self._upload_and_publish_environment(existing_env_id, env_yml_content)
                return existing_env_id

            print("❌ Environment name is already in use, but existing item ID could not be resolved")
            print(response.text)
            sys.exit(1)
        else:
            print(f"❌ Failed to create environment: {response.status_code}")
            print(response.text)
            sys.exit(1)
    
    def create_spark_job(self, name: str, environment_id: str, src_dir: str = "src") -> str:
        """
        Create a Fabric Spark Job Definition.
        
        Args:
            name: Name of the Spark Job Definition
            environment_id: ID of the environment to use
            
        Returns:
            Spark Job Definition ID
        """
        print(f"\n⚡ Creating Spark Job Definition: {name}")

        src_path = self._resolve_source_dir(src_dir)
        library_files = sorted([p.name for p in src_path.glob("*.py") if p.name != "main.py"])

        # Build the job configuration
        job_config = {
            "executableFile": "main.py",
            "defaultLakehouseArtifactId": self.lakehouse_id,
            "mainClass": "",
            "additionalLakehouseIds": [],
            "retryPolicy": None,
            "commandLineArguments": "",
            "additionalLibraryUris": library_files,
            "language": "Python",
            "environmentArtifactId": environment_id
        }
        print(f"job_config:{job_config}")
        parts = self._build_definition_parts(
            job_config=job_config,
            src_path=src_path,
        )

        create_or_update_payload = {
            "displayName": name,
            "type": "SparkJobDefinition",
            "definition": {
                "format": "SparkJobDefinitionV2",
                "parts": parts
            }
        }

        # Create Spark Job Definition item
        url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/items"
        response = requests.post(url, headers=self._get_headers(), json=create_or_update_payload)

        # Update existing Spark Job Definition when name already exists
        if response.status_code == 409:
            existing_job_id = None
            for _ in range(24):
                existing_job_id = self._find_existing_item_id(name, "SparkJobDefinition")
                if existing_job_id:
                    break
                time.sleep(5)
            if not existing_job_id:
                print("❌ Spark Job Definition name is already in use, but existing item ID could not be resolved")
                print(response.text)
                sys.exit(1)
            job_id = existing_job_id
            print(f"ℹ️ Spark Job Definition already exists, updating: {job_id}")

            update_url = f"{self.FABRIC_API_BASE}/workspaces/{self.workspace_id}/items/{job_id}"
            update_response = requests.post(update_url, headers=self._get_headers(), json=create_or_update_payload)
            if update_response.status_code not in [200, 201, 202]:
                print(f"❌ Failed to update Spark Job Definition: {update_response.status_code}")
                print(update_response.text)
                sys.exit(1)
        elif response.status_code not in [200, 201, 202]:
            print(f"❌ Failed to create Spark Job Definition: {response.status_code}")
            print(response.text)
            sys.exit(1)
        else:
            job_id = None
            try:
                job_data = response.json()
            except ValueError:
                job_data = None

            if isinstance(job_data, dict):
                job_id = job_data.get("id")

            if not job_id:
                for _ in range(15):
                    job_id = self._find_existing_item_id(name, "SparkJobDefinition")
                    if job_id:
                        break
                    time.sleep(2)
            if not job_id:
                print("❌ Spark Job Definition create request accepted, but item ID could not be resolved")
                sys.exit(1)
            print(f"✅ Spark Job Definition created: {job_id}")

        print("✅ Spark Job Definition configured successfully")
        return job_id
    
    def deploy(
        self,
        env_name: str = "ts-forecasting-env",
        job_name: str = "e2e-ml-pipeline-spark-job",
        src_dir: str = "src",
    ):
        """
        Deploy both environment and Spark Job Definition.
        
        Args:
            env_name: Name for the environment
            job_name: Name for the Spark Job Definition
        """
        print("=" * 70)
        print("DEPLOYING TO MICROSOFT FABRIC")
        print("=" * 70)
        print(f"Workspace ID: {self.workspace_id}")
        print(f"Lakehouse ID: {self.lakehouse_id}")
        print("=" * 70)
        
        # Authenticate
        self.authenticate()
        
        # Create environment
        env_id = self.create_environment(env_name)
        
        # Create Spark Job Definition
        job_id = self.create_spark_job(
            job_name,
            env_id,
            src_dir=src_dir,
        )
        
        print("\n" + "=" * 70)
        print("✅ DEPLOYMENT COMPLETE")
        print("=" * 70)
        print(f"Environment ID: {env_id}")
        print(f"Spark Job ID: {job_id}")
        print("\nNext steps:")
        print("1. Wait for the environment to finish publishing (~5 minutes)")
        print("2. Local files were uploaded into the Spark Job Definition")
        print("3. Job runs using packaged files (no Lakehouse Files/src dependency)")
        print(f"4. Open the Spark Job Definition '{job_name}' in Fabric")
        print("5. Click 'Run' to execute the pipeline")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Environment and Spark Job Definition to Microsoft Fabric"
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Microsoft Fabric workspace ID"
    )
    parser.add_argument(
        "--lakehouse-id",
        required=True,
        help="Lakehouse ID to attach as default"
    )
    parser.add_argument(
        "--environment-name",
        default="ts-forecasting-env",
        help="Name for the environment (default: ts-forecasting-env)"
    )
    parser.add_argument(
        "--job-name",
        default="fabric-e2e-demo-ml-pipeline-job",
        help="Name for the Spark Job Definition (default: e2e-ml-pipeline-spark-job)"
    )
    parser.add_argument(
        "--src-dir",
        default="src",
        help="Local source folder containing *.py files (default: ./src)"
    )
    
    args = parser.parse_args()
    
    # Create deployer and deploy
    deployer = FabricDeployer(args.workspace_id, args.lakehouse_id)
    deployer.deploy(
        args.environment_name,
        args.job_name,
        src_dir=args.src_dir,
    )


if __name__ == "__main__":
    main()
