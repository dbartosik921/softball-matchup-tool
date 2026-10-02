"use server";
import { redirect } from "next/navigation";
import { checkPassword, endSession, startSession } from "@/lib/auth";

export async function login(_: string | null, form: FormData): Promise<string | null> {
  const pw = String(form.get("password") ?? "");
  if (!checkPassword(pw)) {
    await new Promise((r) => setTimeout(r, 600)); // slow down guessing
    return "Wrong password.";
  }
  await startSession();
  redirect("/");
}

export async function logout(): Promise<void> {
  await endSession();
  redirect("/login");
}
