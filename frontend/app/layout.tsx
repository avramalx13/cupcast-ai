import type { Metadata } from "next";
import { DataModeBadge } from "@/components/DataModeBadge";
import "./globals.css";

export const metadata: Metadata = {
  title: "CupCast AI",
  description: "World Cup prediction engine with a custom mini-LLM analyst"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <DataModeBadge />
        {children}
      </body>
    </html>
  );
}
