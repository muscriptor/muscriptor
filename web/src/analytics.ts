/**
 * Thin Google Analytics (GA4) wrapper.
 *
 * The measurement ID comes from VITE_GA_MEASUREMENT_ID at build time; it is
 * only set for the muscriptor.kyutai.org deployment (a build arg in swarm.yml),
 * so local dev, the PyPI package, and self-hosted builds have analytics fully
 * disabled — `track` is a no-op and the Google script is never loaded.
 */

const GA_ID: string | undefined = import.meta.env.VITE_GA_MEASUREMENT_ID;

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

/** Load gtag.js and configure the property. Call once at startup. */
export function initAnalytics() {
  if (!GA_ID) return;
  window.dataLayer = window.dataLayer ?? [];
  // gtag.js requires the raw `arguments` object to be pushed, not an array.
  window.gtag = function () {
    // eslint-disable-next-line prefer-rest-params
    window.dataLayer!.push(arguments);
  };
  window.gtag("js", new Date());
  window.gtag("config", GA_ID);
  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA_ID)}`;
  document.head.appendChild(script);
}

export type EventParams = Record<string, string | number | boolean | undefined>;

/** Send a custom event. No-op when analytics is disabled.
 *
 * GA4 silently drops event params whose string value exceeds 100 characters,
 * so long values (e.g. a big instrument list) are truncated instead. */
export function track(event: string, params: EventParams = {}) {
  if (!window.gtag) return;
  const cleaned: EventParams = {};
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined) continue;
    cleaned[k] = typeof v === "string" && v.length > 100 ? v.slice(0, 99) + "…" : v;
  }
  window.gtag("event", event, cleaned);
}
