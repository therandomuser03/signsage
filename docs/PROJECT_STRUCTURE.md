# SignSage — Final Project Structure

```
SignSage/
│
├── frontend/                                      # React + TypeScript + Vite
│   │
│   ├── public/
│   │   ├── icons/
│   │   ├── images/
│   │   │   ├── brand/
│   │   │   ├── landing/
│   │   │   ├── history/
│   │   │   └── technology/
│   │   ├── models/
│   │   │   └── avatar/
│   │   │       └── manifest.json
│   │   └── manifest.webmanifest
│   │
│   ├── src/
│   │   │
│   │   ├── app/                                  # Application composition
│   │   │   ├── App.tsx
│   │   │   ├── routes.tsx
│   │   │   ├── providers.tsx
│   │   │   └── config.ts
│   │   │
│   │   ├── pages/                                # Route-level pages
│   │   │   │
│   │   │   ├── landing/
│   │   │   │   ├── LandingPage.tsx
│   │   │   │   ├── sections/
│   │   │   │   │   ├── HeroSection.tsx
│   │   │   │   │   ├── ProductPreviewSection.tsx
│   │   │   │   │   ├── HowItWorksSection.tsx
│   │   │   │   │   ├── CapabilitiesSection.tsx
│   │   │   │   │   ├── TechnologySection.tsx
│   │   │   │   │   ├── ProjectHistorySection.tsx
│   │   │   │   │   ├── CreditsSection.tsx
│   │   │   │   │   ├── PrivacySection.tsx
│   │   │   │   │   ├── FAQSection.tsx
│   │   │   │   │   └── CTASection.tsx
│   │   │   │   ├── components/
│   │   │   │   │   ├── LandingNavbar.tsx
│   │   │   │   │   ├── FloatingNavbar.tsx
│   │   │   │   │   ├── ProductMockup.tsx
│   │   │   │   │   └── ScrollProgress.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── auth/
│   │   │   │   ├── LoginPage.tsx
│   │   │   │   ├── SignupPage.tsx
│   │   │   │   ├── ForgotPasswordPage.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── interpreter/
│   │   │   │   ├── InterpreterPage.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── history/
│   │   │   │   ├── HistoryPage.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── settings/
│   │   │   │   ├── SettingsPage.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── legal/
│   │   │   │   ├── PrivacyPage.tsx
│   │   │   │   ├── TermsPage.tsx
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   └── errors/
│   │   │       ├── NotFoundPage.tsx
│   │   │       └── ErrorPage.tsx
│   │   │
│   │   ├── features/                             # Product/domain features
│   │   │   │
│   │   │   ├── authentication/
│   │   │   │   ├── components/
│   │   │   │   ├── hooks/
│   │   │   │   ├── services/
│   │   │   │   ├── types/
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── translation/
│   │   │   │   ├── components/
│   │   │   │   ├── hooks/
│   │   │   │   ├── services/
│   │   │   │   ├── state/
│   │   │   │   ├── types/
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── sign-language/
│   │   │   │   ├── components/
│   │   │   │   │   ├── CameraCapture.tsx
│   │   │   │   │   ├── LandmarkOverlay.tsx
│   │   │   │   │   ├── SignPreview.tsx
│   │   │   │   │   └── SignStatus.tsx
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── useCamera.ts
│   │   │   │   │   ├── useLandmarks.ts
│   │   │   │   │   └── useSignCapture.ts
│   │   │   │   ├── services/
│   │   │   │   │   └── mediapipe.ts
│   │   │   │   ├── processing/
│   │   │   │   │   ├── landmarks.ts
│   │   │   │   │   ├── frame-buffer.ts
│   │   │   │   │   └── motion.ts
│   │   │   │   ├── types/
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── text-to-sign/
│   │   │   │   ├── components/
│   │   │   │   │   └── AvatarViewer.tsx
│   │   │   │   ├── hooks/
│   │   │   │   │   └── useSignAnimation.ts
│   │   │   │   ├── services/
│   │   │   │   │   ├── avatarController.ts
│   │   │   │   │   └── glossMapper.ts
│   │   │   │   ├── state/
│   │   │   │   ├── types/
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── history/
│   │   │   │   ├── components/
│   │   │   │   ├── hooks/
│   │   │   │   ├── services/
│   │   │   │   ├── types/
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   └── settings/
│   │   │       ├── components/
│   │   │       ├── hooks/
│   │   │       ├── services/
│   │   │       ├── types/
│   │   │       └── index.ts
│   │
│   │   ├── components/                           # Truly shared UI
│   │   │   ├── ui/                               # shadcn/base primitives
│   │   │   ├── layout/
│   │   │   │   ├── PublicLayout.tsx
│   │   │   │   ├── AppLayout.tsx
│   │   │   │   └── AuthLayout.tsx
│   │   │   ├── navigation/
│   │   │   │   ├── AppNavbar.tsx
│   │   │   │   ├── AppSidebar.tsx
│   │   │   │   └── UserMenu.tsx
│   │   │   ├── feedback/
│   │   │   └── common/
│   │   │
│   │   ├── services/                             # Cross-feature infrastructure
│   │   │   ├── api/
│   │   │   │   ├── client.ts
│   │   │   │   ├── instance.ts
│   │   │   │   ├── endpoints.ts
│   │   │   │   └── errors.ts
│   │   │   ├── websocket/
│   │   │   │   ├── client.ts
│   │   │   │   ├── connection.ts
│   │   │   │   └── protocol.ts
│   │   │   └── supabase/
│   │   │       ├── client.ts
│   │   │       └── auth.ts
│   │   │
│   │   ├── hooks/                                # Generic reusable hooks
│   │   │
│   │   ├── lib/                                  # Generic utilities
│   │   │   ├── mediapipe/
│   │   │   ├── storage/
│   │   │   ├── validation/
│   │   │   └── utils/
│   │   │
│   │   ├── state/                                # Truly global state
│   │   │   └── ...
│   │   │
│   │   ├── types/                                # Global/shared types
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   └── common.ts
│   │   │
│   │   ├── styles/
│   │   │   ├── globals.css
│   │   │   ├── animations.css
│   │   │   └── landing.css
│   │   │
│   │   └── main.tsx
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   ├── .env.example
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── vitest.config.ts
│   └── playwright.config.ts
│
│
├── backend/                                      # Python + FastAPI
│   │
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py
│   │   │   │   ├── auth.py
│   │   │   │   ├── translation.py
│   │   │   │   └── history.py
│   │   │   ├── websocket/
│   │   │   │   └── translation.py
│   │   │   └── router.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   ├── errors.py
│   │   │   ├── security.py
│   │   │   └── dependencies.py
│   │   │
│   │   ├── domain/
│   │   │   ├── translation/
│   │   │   │   ├── entities.py
│   │   │   │   ├── value_objects.py
│   │   │   │   └── policies.py
│   │   │   ├── sign_language/
│   │   │   │   ├── entities.py
│   │   │   │   └── contracts.py
│   │   │   └── users/
│   │   │       └── entities.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── common.py
│   │   │   ├── auth.py
│   │   │   ├── translation.py
│   │   │   ├── websocket.py
│   │   │   └── history.py
│   │   │
│   │   ├── services/
│   │   │   ├── translation_service.py
│   │   │   ├── session_service.py
│   │   │   ├── history_service.py
│   │   │   └── authentication_service.py
│   │   │
│   │   ├── ml/
│   │   │   ├── preprocessing/
│   │   │   │   ├── landmarks.py
│   │   │   │   ├── normalization.py
│   │   │   │   └── sequence.py
│   │   │   ├── inference/
│   │   │   │   ├── sign_to_text.py
│   │   │   │   └── text_to_sign.py
│   │   │   ├── models/
│   │   │   │   ├── sign_to_text.py
│   │   │   │   ├── text_to_sign.py
│   │   │   │   └── registry.py
│   │   │   ├── adapters/
│   │   │   │   ├── sign_to_text_adapter.py
│   │   │   │   └── text_to_sign_adapter.py
│   │   │   ├── postprocessing/
│   │   │   │   ├── decoding.py
│   │   │   │   ├── confidence.py
│   │   │   │   └── refinement.py
│   │   │   └── contracts/
│   │   │       ├── landmark_contract.py
│   │   │       ├── model_contract.py
│   │   │       └── inference_contract.py
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── database/
│   │   │   │   ├── supabase.py
│   │   │   │   ├── repositories/
│   │   │   │   │   └── translation_repository.py
│   │   │   │   └── migrations/
│   │   │   ├── storage/
│   │   │   │   └── model_storage.py
│   │   │   └── external/
│   │   │       └── ...
│   │   │
│   │   └── main.py
│   │
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── domain/
│   │   │   ├── services/
│   │   │   └── ml/
│   │   ├── integration/
│   │   └── contract/
│   │
│   ├── scripts/
│   │   ├── health_check.py
│   │   ├── model_check.py
│   │   └── ...
│   │
│   ├── models/
│   │   └── README.md
│   │
│   ├── .env.example
│   ├── pyproject.toml
│   └── Dockerfile
│
│
├── research/                                     # Training / experimentation only
│   ├── notebooks/
│   │   └── signsage/
│   │       └── signsage_latest.ipynb
│   ├── configs/
│   │   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   ├── datasets/
│   │   └── README.md
│   ├── experiments/
│   │   ├── sign-to-text/
│   │   └── text-to-sign/
│   ├── evaluation/
│   │   ├── metrics/
│   │   ├── reports/
│   │   └── benchmarks/
│   ├── artifacts/
│   │   └── README.md
│   └── manifests/
│       ├── datasets/
│       ├── experiments/
│       └── models/
│
│
├── docs/                                         # Project source of truth
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── frontend.md
│   │   ├── backend.md
│   │   ├── data-flow.md
│   │   └── boundaries.md
│   │
│   ├── frontend/
│   │   ├── architecture.md
│   │   ├── features.md
│   │   ├── pages.md
│   │   ├── state-management.md
│   │   ├── mediapipe.md
│   │   └── development.md
│   │
│   ├── backend/
│   │   ├── architecture.md
│   │   ├── services.md
│   │   ├── error-handling.md
│   │   └── development.md
│   │
│   ├── api/
│   │   ├── overview.md
│   │   ├── http.md
│   │   ├── websocket.md
│   │   ├── schemas.md
│   │   └── versioning.md
│   │
│   ├── ai-ml/
│   │   ├── overview.md
│   │   ├── sign-to-text.md
│   │   ├── text-to-sign.md
│   │   ├── preprocessing.md
│   │   ├── model-contract.md
│   │   ├── model-registry.md
│   │   └── research-to-production.md
│   │
│   ├── database/
│   │   ├── overview.md
│   │   ├── schema.md
│   │   ├── authentication.md
│   │   └── authorization.md
│   │
│   ├── development/
│   │   ├── setup.md
│   │   ├── coding-conventions.md
│   │   ├── testing.md
│   │   ├── debugging.md
│   │   └── ai-agent-guide.md
│   │
│   ├── deployment/
│   │   ├── architecture.md
│   │   ├── frontend.md
│   │   ├── backend.md
│   │   ├── environment.md
│   │   └── rollback.md
│   │
│   ├── troubleshooting/
│   │   ├── common-issues.md
│   │   ├── camera.md
│   │   ├── mediapipe.md
│   │   ├── websocket.md
│   │   ├── inference.md
│   │   └── deployment.md
│   │
│   ├── decisions/
│   │   ├── README.md
│   │   ├── ADR-001-frontend-framework.md
│   │   ├── ADR-002-backend-framework.md
│   │   ├── ADR-003-websocket-transport.md
│   │   ├── ADR-004-model-runtime-boundary.md
│   │   └── ...
│   │
│   └── security/
│       ├── overview.md
│       ├── authentication.md
│       ├── data-privacy.md
│       └── secrets.md
│
│
├── infrastructure/                                # Deployment / operations
│   ├── docker/
│   │   ├── frontend/
│   │   └── backend/
│   ├── deployment/
│   │   ├── vercel/
│   │   └── render/
│   └── monitoring/
│       └── README.md
│
│
├── AGENTS.md
├── CONTRIBUTING.md
├── README.md
├── LICENSE
├── .gitignore
└── .env.example
```
