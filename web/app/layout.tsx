import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpatialSight",
  description:
    "Pixel-level video understanding compiled into a relational schema you can query in plain English.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
