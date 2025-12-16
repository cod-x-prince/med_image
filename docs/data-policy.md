# Data Governance & Security Policy

## 1. Medical Data Handling

- **Zero Retention**: HemaVision AI is designed to process medical imaging data (DICOM, NIfTI, PNG/JPG) strictly in-memory.
- **No Persistence**: Uploaded files are processed and immediately discarded. No original patient data is stored in databases or file systems.
- **Anonymization**: Any testing data used during development must be fully anonymized (de-identified) compliant with HIPAA/GDPR standards.

## 2. Secrets Management

- **No Hardcoded Secrets**: API keys, database credentials, and secret keys must be loaded via environment variables or a secrets manager.
- **Public Repository Safety**: The `.env` file is stricly ignored. No real credentials are committed to version control.

## 3. Compliance

- **Audit Logging**: The system maintains an audit log of processing events (timestamps, status) without recording PII (Personal Identifiable Information).
- **Access Control**: Future deployments should implement Role-Based Access Control (RBAC).

## 4. Exclusion List

The following are strictly forbidden in this repository:

- `*.dcm`, `*.nii`, `*.nii.gz`
- Patient filenames or IDs
- Model checkpoints trained on private data (unless weights are public/licensed)
