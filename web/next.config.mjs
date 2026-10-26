/** @type {import('next').NextConfig} */

// The Python process owns the database and the SQL guard; the UI only renders.
// Proxying keeps the browser on one origin so there is no CORS to configure.
const API = process.env.SPATIALSIGHT_API ?? "http://127.0.0.1:8000";

const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/media/:path*", destination: `${API}/media/:path*` },
    ];
  },
};

export default nextConfig;
