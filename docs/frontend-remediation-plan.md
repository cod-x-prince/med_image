# Frontend Remediation Plan

## Current State

The application currently utilizes server-side rendering with Flask (Jinja2 templates) and vanilla JavaScript/CSS. The `viewer_prototype.html` represents an early-stage 3D visualization component.

## Architecture Evolution

We aim to transition from a Monolithic architecture to a Decoupled architecture.

### Phase 1: Modularization (Current)

- Standardize `templates` and `static` assets.
- Ensure clear separation of concerns in CSS/JS.
- Isolate the 3D viewer component.

### Phase 2: React/Next.js Migration

- Rebuild the dashboard using a modern framework (React or Next.js).
- Consume the Flask backend purely as an API (REST/GraphQL).
- Implement a dedicated DICOM viewer (e.g., OHIF or Cornerstone.js integration) on the client side.

### Phase 3: State Management

- Migrate session handling to client-side JWT or secure cookies.
- Remove dependency on server-side session storage for UI state.
