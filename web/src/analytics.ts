/**
 * Thin Google Analytics (GA4) wrapper, gated behind cookie consent.
 *
 * The measurement ID comes from VITE_GA_MEASUREMENT_ID at build time; it is
 * only set for the muscriptor.kyutai.org deployment (a build arg in swarm.yml),
 * so local dev, the PyPI package, and self-hosted builds have analytics fully
 * disabled — `track` is a no-op and the Google script is never loaded.
 *
 * Even in that deployment, gtag.js only loads after the user accepts the
 * consent banner: until then — or forever, after a decline — nothing is sent
 * and no cookies are set.
 */

const GA_ID: string | undefined = import.meta.env.VITE_GA_MEASUREMENT_ID;

const CONSENT_KEY = "cookieConsent";

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

let loaded = false;

function loadGtag() {
  if (loaded || !GA_ID) return;
  loaded = true;
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

/** True when this build has a measurement ID, i.e. analytics (and therefore
 *  the consent banner) is relevant at all. */
export function analyticsAvailable(): boolean {
  return Boolean(GA_ID);
}

/** The consent choice saved in a previous visit; null = not chosen yet
 *  (show the banner). Treated as declined when localStorage is unavailable. */
export function storedConsent(): boolean | null {
  try {
    const v = localStorage.getItem(CONSENT_KEY);
    return v == null ? null : v === "true";
  } catch {
    return false;
  }
}

/** Persist the user's choice; accepting starts analytics immediately. */
export function setConsent(consent: boolean) {
  try {
    localStorage.setItem(CONSENT_KEY, String(consent));
  } catch {
    // Storage unavailable — the choice still applies for this page load.
  }
  if (consent) loadGtag();
}

/** Call once at startup: starts analytics if consent was already given. */
export function initAnalytics() {
  if (storedConsent() === true) loadGtag();
}

export type EventParams = Record<string, string | number | boolean | undefined>;

/** Send a custom event. No-op until analytics is loaded (build has a
 * measurement ID + user consented), so call sites never need to check.
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
