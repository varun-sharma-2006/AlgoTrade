import { Fragment, type ReactNode } from "react";

/**
 * Renders the small subset of Markdown that LLM replies use (**bold**, bullets, # headings)
 * as React elements. No HTML is injected, so model output can't smuggle markup into the page.
 */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={index}>{part.slice(2, -2)}</strong>
    ) : (
      <Fragment key={index}>{part.replace(/(^|\s)\*(\S[^*]*\S|\S)\*(?=\s|$|[.,;:!?])/g, "$1$2")}</Fragment>
    ),
  );
}

export function FormattedText({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <>
      {lines.map((raw, index) => {
        const heading = raw.match(/^\s*#{1,6}\s+(.*)$/);
        const bullet = raw.match(/^(\s*)[*-]\s+(.*)$/);
        const content = heading ? <strong>{inline(heading[1])}</strong> : bullet ? (
          <>
            {bullet[1]}• {inline(bullet[2])}
          </>
        ) : (
          inline(raw)
        );
        return (
          <Fragment key={index}>
            {content}
            {index < lines.length - 1 ? "\n" : null}
          </Fragment>
        );
      })}
    </>
  );
}
