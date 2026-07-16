/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** GA4 measurement ID; unset = analytics disabled (see src/analytics.ts). */
  readonly VITE_GA_MEASUREMENT_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
