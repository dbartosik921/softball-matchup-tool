import "server-only";
import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

const COOKIE = "matchup_session";
const DAYS = 30;

function secret(): string {
  const s = process.env.SESSION_SECRET || process.env.ADMIN_PASSWORD;
  if (!s) throw new Error("ADMIN_PASSWORD is not set");
  return createHash("sha256").update("matchup-session:" + s).digest("hex");
}

function sign(exp: number): string {
  return createHmac("sha256", secret()).update(String(exp)).digest("base64url");
}

function same(a: string, b: string): boolean {
  const x = createHash("sha256").update(a).digest();
  const y = createHash("sha256").update(b).digest();
  return timingSafeEqual(x, y);
}

export function passwordConfigured(): boolean {
  return Boolean(process.env.ADMIN_PASSWORD);
}

export function checkPassword(given: string): boolean {
  const want = process.env.ADMIN_PASSWORD;
  return Boolean(want) && same(given, want!);
}

export async function startSession(): Promise<void> {
  const exp = Date.now() + DAYS * 864e5;
  (await cookies()).set(COOKIE, `${exp}.${sign(exp)}`, {
    httpOnly: true, secure: process.env.NODE_ENV === "production", sameSite: "lax", path: "/",
    maxAge: DAYS * 86400,
  });
}

export async function endSession(): Promise<void> {
  (await cookies()).delete(COOKIE);
}

export async function isAdmin(): Promise<boolean> {
  const v = (await cookies()).get(COOKIE)?.value;
  if (!v || !passwordConfigured()) return false;
  const [exp, sig] = v.split(".");
  const n = Number(exp);
  if (!n || n < Date.now() || !sig) return false;
  return same(sig, sign(n));
}

/** Call at the top of every page / route that shows data. */
export async function requireAdmin(): Promise<void> {
  if (!(await isAdmin())) redirect("/login");
}
