import { NextResponse } from "next/server";
import { isAdmin } from "@/lib/auth";
import { latestRun, team } from "@/lib/data";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  if (!(await isAdmin())) return NextResponse.json({ error: "sign in" }, { status: 401 });
  const code = new URL(req.url).searchParams.get("team") ?? "";
  const run = await latestRun();
  const t = run && code ? await team(run.run_id, code) : null;
  return NextResponse.json((t?.roster ?? []).map(({ id, name, pa }) => ({ id, name, pa })));
}
