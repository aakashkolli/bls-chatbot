import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'BLS Virtual Advisor Widget',
  description: 'Embeddable BLS advising chat widget'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
