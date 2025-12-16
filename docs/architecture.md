# HemaVision AI - System Architecture

## Overview

HemaVision AI is a medical imaging analysis platform designed to assist in the detection and classification of anomalies in medical scans.

## Core Components

### 1. Application Layer (`src/app`)

- **Framework**: Flask (Python)
- **Responsibility**: HTTP API endpoints, Request handling, Input validation, Task dispatching.

### 2. Inference Engine (`src/inference`)

- **Responsibility**: Loading ML models, Pre-processing images, Running inference, Post-processing results.
- **Integrations**: TensorFlow / PyTorch / Gemini API.
- **3D Reconstruction**: Point-cloud generation and mesh rendering pipeline.

### 3. Data Processing (`src/utils`)

- **DICOM Handling**: Parsing metadata, pixel data extraction using `pydicom`.
- **Normalization**: Image standardization logic.

### 4. Background Tasks (Planned)

- **Celery**: For long-running inference tasks (Video processing, 3D reconstruction).
- **Redis**: Message broker.

### 5. Infrastructure

- **Docker**: Containerization for reproducible environments.
- **Database**: Sqlite (Dev) / Postgres (Prod) for storing _metadata only_ (Audit logs, User accounts). NO IMAGE DATA.

## Data Flow

1.  **Upload**: User uploads image via API.
2.  **Validate**: Server verifies format and compliance.
3.  **Process**: Image passed to Inference Engine (RAM).
4.  **Result**: Analysis returned to User.
5.  **Cleanup**: Image data wiped from memory/temp storage.
