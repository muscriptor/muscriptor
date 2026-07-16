import { useState } from "react";
import { Button } from "./Button";
import { analyticsAvailable, setConsent, storedConsent } from "../analytics";

/**
 * Bottom-of-page cookie consent bar: Google Analytics only ever loads after
 * an explicit "Accept", and the choice is remembered in localStorage. Renders
 * nothing in builds without a measurement ID (where there is no analytics to
 * consent to) or once a choice was made.
 */
export function ConsentBanner() {
  const [answered, setAnswered] = useState(
    () => !analyticsAvailable() || storedConsent() !== null,
  );
  if (answered) return null;

  const choose = (consent: boolean) => {
    setConsent(consent);
    setAnswered(true);
  };

  return (
    <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-surface">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 gap-y-3 px-7 py-4">
        <p className="max-w-2xl text-sm text-muted">
          We use cookies (Google Analytics) to understand how this demo is
          used: page visits and anonymous usage events, e.g. the length of the
          transcribed audio and the instruments you select; never the audio
          itself. See the{" "}
          <a
            href="https://kyutai.org/privacy-policy"
            target="_blank"
            rel="noreferrer"
            className="text-accent underline underline-offset-4 opacity-90 hover:opacity-100"
          >
            privacy policy
          </a>
          .
        </p>
        <div className="ml-auto flex gap-2.5">
          <Button onClick={() => choose(true)}>Accept</Button>
          <Button kind="secondaryOff" onClick={() => choose(false)}>
            Decline
          </Button>
        </div>
      </div>
    </div>
  );
}
