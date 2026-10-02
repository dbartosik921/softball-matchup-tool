import Link from "next/link";
import { logout } from "@/app/actions";

export default function Header({ through }: { through?: string }) {
  return (
    <header className="top">
      <Link className="brand" href="/">Matchups</Link>
      <div className="right">
        {through && <span>Data through {through}</span>}
        <form action={logout}><button className="link">Sign out</button></form>
      </div>
    </header>
  );
}
