import { Fira_Code as FontMono, Inter as FontSans, Montserrat as FontBody } from "next/font/google";

export const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const fontBody = FontBody({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-body",
});

export const fontMono = FontMono({
  subsets: ["latin"],
  variable: "--font-mono",
});
