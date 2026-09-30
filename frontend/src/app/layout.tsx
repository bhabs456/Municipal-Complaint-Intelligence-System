import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Municipal Complaint Intelligence System",
  description: "AI-driven municipal complaint tracking and intelligence portal",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
