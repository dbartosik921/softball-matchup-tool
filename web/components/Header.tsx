import Link from "next/link";
import { logout } from "@/app/actions";

export default function Header({ through, tab }: { through?: string; tab?: "matchup" | "gameday" }) {
  return (
    <header className="top">
      <nav className="tabs">
        <Link className="brand" href="/">Matchups</Link>
        <Link className={tab === "matchup" ? "on" : ""} href="/">Matchup</Link>
        <Link className={tab === "gameday" ? "on" : ""} href="/gameday">Gameday</Link>
      </nav>
      <div className="right">
        {through && <span>Data through {through}</span>}
        <form action={logout}><button className="link">Sign out</button></form>
      </div>
    </header>
  );
}
