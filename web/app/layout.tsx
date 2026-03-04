import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });

export const metadata: Metadata = {
  title: "March Madness AI — ML Bracket Predictor",
  description:
    "Machine-learning bracket predictions for NCAA March Madness. Trained on 21 seasons of tournament data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={geist.variable}>
      <body className="min-h-screen bg-navy antialiased">
        <Nav />
        <main>{children}</main>
        <footer className="border-t border-slate-800 mt-16 py-8 text-center text-slate-600 text-sm">
          <p>
            Stacked ensemble model · 21 seasons of NCAA data · 69.8% accuracy (season-aware CV)
          </p>
          <p className="mt-1">
            Built with{" "}
            <span className="text-orange">scikit-learn</span> +{" "}
            <span className="text-orange">Next.js</span>
          </p>
        </footer>
      </body>
    </html>
  );
}
