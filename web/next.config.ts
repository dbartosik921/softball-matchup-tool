import type { NextConfig } from "next";

const config: NextConfig = {
  poweredByHeader: false,
  async headers() {
    // Private site with player data: never cached by shared caches, never indexed.
    return [{
      source: "/:path*",
      headers: [
        { key: "X-Robots-Tag", value: "noindex, nofollow" },
        { key: "Cache-Control", value: "private, no-store" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Referrer-Policy", value: "same-origin" },
      ],
    }];
  },
};
export default config;
