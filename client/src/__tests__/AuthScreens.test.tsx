import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LoginForm } from "../components/LoginForm";
import { SignupForm } from "../components/SignupForm";

afterEach(cleanup);

describe("sign-in screens", () => {
  it("shows the split layout and validates the sign-up form", () => {
    const onSubmit = vi.fn();
    render(<SignupForm onSubmit={onSubmit} onSwitchToLogin={vi.fn()} loading={false} />);
    expect(screen.getByRole("heading", { name: /Turn market ideas into measured/ })).toBeInTheDocument();
    expect(screen.getByText("At least 8 characters. Very common passwords are rejected.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ravi" } });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "ravi@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));
    expect(screen.getByText("Password must be at least 8 characters long.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "tide-pool-ledger" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));
    expect(onSubmit).toHaveBeenCalledWith("Ravi", "ravi@example.com", "tide-pool-ledger");
  });

  it("switches from sign-in to sign-up", () => {
    const onSwitch = vi.fn();
    render(<LoginForm onSubmit={vi.fn()} onSwitchToSignup={onSwitch} loading={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Create an account" }));
    expect(onSwitch).toHaveBeenCalled();
  });
});

describe("API error text", () => {
  it("turns validation errors into sentences", async () => {
    const { describeDetail } = await import("../api");
    expect(describeDetail([{ msg: "Value error, That password is too common" }, { msg: "Field required" }])).toBe(
      "That password is too common. Field required",
    );
    expect(describeDetail("Plain message")).toBe("Plain message");
  });
});
