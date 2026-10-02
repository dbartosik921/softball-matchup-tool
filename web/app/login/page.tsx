import { redirect } from "next/navigation";
import { isAdmin, passwordConfigured } from "@/lib/auth";
import LoginForm from "./form";

export const dynamic = "force-dynamic";

export default async function Login() {
  if (await isAdmin()) redirect("/");
  return (
    <main className="login">
      <h1>Matchups</h1>
      {passwordConfigured()
        ? <LoginForm />
        : <p className="err">ADMIN_PASSWORD isn&apos;t set for this deployment. Add it in Vercel → Settings → Environment Variables, then redeploy.</p>}
    </main>
  );
}
