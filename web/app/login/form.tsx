"use client";
import { useActionState } from "react";
import { login } from "../actions";

export default function LoginForm() {
  const [error, action, pending] = useActionState(login, null);
  return (
    <form action={action} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <label>Password<input type="password" name="password" autoComplete="current-password" autoFocus required /></label>
      <button disabled={pending}>{pending ? "Signing in…" : "Sign in"}</button>
      {error && <p className="err">{error}</p>}
    </form>
  );
}
