"""Supabase storage service for document files."""

from typing import Optional
from uuid import UUID

from supabase import create_client, Client

from app.config import settings


class StorageService:
    """Supabase storage service for document files."""

    def __init__(self):
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        """Get or create Supabase client."""
        if not self._client:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
            )
        return self._client

    def _get_file_path(self, org_id: UUID, document_id: UUID, filename: str) -> str:
        """Generate storage file path."""
        return f"{org_id}/{document_id}/{filename}"

    async def upload_file(
        self,
        org_id: UUID,
        document_id: UUID,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload a file to Supabase storage.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename
            content: File content bytes
            content_type: MIME type

        Returns:
            Storage file path
        """
        client = self._get_client()
        file_path = self._get_file_path(org_id, document_id, filename)

        # Upload to storage bucket
        client.storage.from_(settings.storage_bucket).upload(
            file_path,
            content,
            file_options={"content-type": content_type},
        )

        return file_path

    async def download_file(
        self, org_id: UUID, document_id: UUID, filename: str
    ) -> bytes:
        """Download a file from Supabase storage.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename

        Returns:
            File content bytes
        """
        client = self._get_client()
        file_path = self._get_file_path(org_id, document_id, filename)

        response = client.storage.from_(settings.storage_bucket).download(file_path)
        return response

    async def delete_file(
        self, org_id: UUID, document_id: UUID, filename: str
    ) -> bool:
        """Delete a file from Supabase storage.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename

        Returns:
            True if deleted successfully
        """
        client = self._get_client()
        file_path = self._get_file_path(org_id, document_id, filename)

        try:
            client.storage.from_(settings.storage_bucket).remove([file_path])
            return True
        except Exception:
            return False

    async def delete_document_files(self, org_id: UUID, document_id: UUID) -> bool:
        """Delete all files for a document.

        Args:
            org_id: Organization ID
            document_id: Document ID

        Returns:
            True if deleted successfully
        """
        client = self._get_client()
        folder_path = f"{org_id}/{document_id}"

        try:
            # List all files in the document folder
            files = client.storage.from_(settings.storage_bucket).list(folder_path)

            if files:
                # Delete all files in the folder
                file_paths = [f"{folder_path}/{f['name']}" for f in files]
                client.storage.from_(settings.storage_bucket).remove(file_paths)

            return True
        except Exception:
            return False

    async def get_public_url(
        self, org_id: UUID, document_id: UUID, filename: str
    ) -> str:
        """Get a public URL for a file.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename

        Returns:
            Public URL string
        """
        client = self._get_client()
        file_path = self._get_file_path(org_id, document_id, filename)

        response = client.storage.from_(settings.storage_bucket).get_public_url(
            file_path
        )
        return response

    async def get_signed_url(
        self,
        org_id: UUID,
        document_id: UUID,
        filename: str,
        expires_in: int = 3600,
    ) -> str:
        """Get a signed URL for temporary access.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename
            expires_in: Expiration time in seconds

        Returns:
            Signed URL string
        """
        client = self._get_client()
        file_path = self._get_file_path(org_id, document_id, filename)

        response = client.storage.from_(settings.storage_bucket).create_signed_url(
            file_path, expires_in
        )
        return response["signedURL"]

    async def file_exists(
        self, org_id: UUID, document_id: UUID, filename: str
    ) -> bool:
        """Check if a file exists in storage.

        Args:
            org_id: Organization ID
            document_id: Document ID
            filename: Original filename

        Returns:
            True if file exists
        """
        client = self._get_client()
        folder_path = f"{org_id}/{document_id}"

        try:
            files = client.storage.from_(settings.storage_bucket).list(folder_path)
            return any(f["name"] == filename for f in files)
        except Exception:
            return False

    async def get_org_storage_usage(self, org_id: UUID) -> float:
        """Calculate total storage usage for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Storage usage in MB
        """
        client = self._get_client()

        try:
            # List all files in org folder recursively
            files = client.storage.from_(settings.storage_bucket).list(str(org_id))

            total_bytes = 0
            for item in files:
                if item.get("metadata"):
                    total_bytes += item["metadata"].get("size", 0)

            return total_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0


# Global storage service instance
storage_service = StorageService()
