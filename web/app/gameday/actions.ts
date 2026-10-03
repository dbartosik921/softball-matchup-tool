"use server";
import { isAdmin } from "@/lib/auth";
import { saveCoachNote } from "@/lib/data";

export async function saveNote(batterId: string, note: string): Promise<{ ok: boolean; error?: string }> {
  if (!(await isAdmin())) return { ok: false, error: "Signed out: reload and sign in." };
  if (typeof batterId !== "string" || !batterId || batterId.length > 40) return { ok: false, error: "Bad batter." };
  try {
    await saveCoachNote(batterId, String(note ?? ""));
    return { ok: true };
  } catch {
    return { ok: false, error: "Couldn't save. Run `python -m matchup migrate` on your Mac to add the notes table." };
  }
}
