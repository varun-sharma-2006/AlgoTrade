import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { FormattedText } from "../components/FormattedText";

afterEach(cleanup);

describe("FormattedText", () => {
  it("renders bold, bullets and headings without asterisks", () => {
    const { container } = render(
      <FormattedText text={"## Trade plan\n*   **Entry:** near 230\n- Stop at 216\nPlain *emphasis* and 3 * 4"} />,
    );
    const text = container.textContent ?? "";
    expect(text).toBe("Trade plan\n• Entry: near 230\n• Stop at 216\nPlain emphasis and 3 * 4");
    const bold = Array.from(container.querySelectorAll("strong"), (el) => el.textContent);
    expect(bold).toEqual(["Trade plan", "Entry:"]);
  });

  it("never renders HTML from the reply", () => {
    const { container } = render(<FormattedText text={'<img src=x onerror="alert(1)"> **hi**'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img");
  });
});
